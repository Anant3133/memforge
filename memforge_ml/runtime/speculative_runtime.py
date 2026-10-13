"""
PagedSpeculativeEngine: speculative decoding over a paged KV cache.

A small draft model proposes K tokens; the large target model verifies all K in a
single forward pass and the statistical rejection sampler decides how many to keep.
The output distribution is provably the target model's own -- the draft only
changes how fast tokens arrive, never which tokens are possible.

Paging pays off twice here:

  * the target's KV cache is grown block-at-a-time while speculating K tokens
    ahead, with no contiguous reservation for a lookahead that may be discarded;
  * rejecting j speculative tokens is `cache.rollback(j)` -- moving a write
    cursor. There is no tensor copy, no reallocation, and the physical blocks
    stay owned by the sequence, so a rejection costs O(1) regardless of context
    length. With a contiguous cache this is where speculative decoding either
    pays a memcpy or over-reserves.

Block tables here are driven from Python against the C++ `BlockAllocator`: the
continuous-batching path already exercises the C++ `Scheduler`, while speculation
needs finer-grained growth and rollback than the C++ `Sequence` models.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

import torch
from transformers.cache_utils import DynamicCache

from memforge_ml.runtime.kv_pool import ModelKVSpec, PagedKVPool
from memforge_ml.runtime.paged_cache import PagedCache
from memforge_ml.runtime.sampling import (
    SamplingParams,
    logits_to_probs,
    sample_from_probs,
)

try:
    import memforge_c

    HAS_CPP_ENGINE = True
except ImportError:  # pragma: no cover
    memforge_c = None
    HAS_CPP_ENGINE = False


@dataclass
class SpecTokenEvent:
    index: int
    token_id: int
    text: str
    source: str  # "draft" = proposed and accepted, "resample" = rejected+corrected, "bonus" = free token
    round_index: int


@dataclass
class SpecStats:
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    resampled: int = 0
    bonus: int = 0
    target_forward_passes: int = 0
    draft_forward_passes: int = 0
    rollback_tokens: int = 0
    accepted_per_round: List[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0


class PagedSpeculativeEngine:
    def __init__(
        self,
        target_model,
        target_tokenizer,
        draft_model,
        draft_tokenizer,
        *,
        lookahead_k: int = 4,
        block_size: int = 16,
        num_blocks: Optional[int] = None,
        kv_budget_mb: float = 384.0,
        max_model_len: int = 2048,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if not HAS_CPP_ENGINE:
            raise RuntimeError(
                "The native memforge_c extension is required. "
                "Build it with: python setup.py build_ext --inplace"
            )

        vt = getattr(target_model.config, "vocab_size", None)
        vd = getattr(draft_model.config, "vocab_size", None)
        if vt and vd and vt != vd:
            raise ValueError(
                f"speculative decoding needs a shared vocabulary; target has {vt} "
                f"tokens and draft has {vd}"
            )

        self.target_model = target_model
        self.tokenizer = target_tokenizer
        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer
        self.lookahead_k = lookahead_k
        self.block_size = block_size
        self.max_model_len = max_model_len
        self.device = device or str(next(target_model.parameters()).device)
        self.dtype = dtype or next(target_model.parameters()).dtype

        self.spec = ModelKVSpec.from_config(target_model.config, self.dtype, self.device)
        if num_blocks is None:
            num_blocks = PagedKVPool.blocks_for_budget(self.spec, block_size, kv_budget_mb)
        self.pool = PagedKVPool(self.spec, num_blocks=num_blocks, block_size=block_size)
        self.allocator = memforge_c.BlockAllocator(num_blocks, block_size)

        self.eos_token_ids = self._resolve_eos_ids()

    def _resolve_eos_ids(self) -> set:
        ids = set()
        for candidate in (
            getattr(self.target_model.config, "eos_token_id", None),
            getattr(self.tokenizer, "eos_token_id", None),
        ):
            if isinstance(candidate, int):
                ids.add(candidate)
            elif isinstance(candidate, (list, tuple)):
                ids.update(int(c) for c in candidate)
        return ids

    def describe(self) -> Dict[str, Any]:
        return {
            "target_model": getattr(self.target_model.config, "name_or_path", "unknown"),
            "draft_model": getattr(self.draft_model.config, "name_or_path", "unknown"),
            "lookahead_k": self.lookahead_k,
            "device": self.device,
            "dtype": str(self.dtype),
            "eos_token_ids": sorted(self.eos_token_ids),
            "kv_pool": self.pool.describe(),
        }

    # ------------------------------------------------------------ block table

    def _grow_blocks(self, block_table: List[int], num_tokens: int) -> List[int]:
        """Extend a block table so it covers `num_tokens` logical positions."""
        needed = (num_tokens + self.block_size - 1) // self.block_size
        deficit = needed - len(block_table)
        if deficit > 0:
            if not self.allocator.can_allocate(deficit):
                raise MemoryError(
                    f"KV pool exhausted: need {deficit} more blocks, "
                    f"{self.allocator.get_num_free_blocks()} free"
                )
            block_table.extend(self.allocator.allocate_blocks(deficit))
        return block_table

    # --------------------------------------------------------------- decoding

    def stream(
        self,
        prompt,
        params: Optional[SamplingParams] = None,
    ) -> Generator[SpecTokenEvent, None, Dict[str, Any]]:
        """
        Generate token-by-token, yielding each token as it is confirmed.

        Returns (as the generator's StopIteration value) the same summary dict
        that `generate()` produces.
        """
        params = params or SamplingParams()
        if isinstance(prompt, str):
            prompt_ids = self.tokenizer(prompt, return_tensors=None)["input_ids"]
        else:
            prompt_ids = list(prompt)
        if not prompt_ids:
            raise ValueError("cannot generate from an empty prompt")

        headroom = self.max_model_len - len(prompt_ids)
        if headroom <= 0:
            raise ValueError(
                f"prompt of {len(prompt_ids)} tokens leaves no room under "
                f"max_model_len={self.max_model_len}"
            )
        params = params.clamped(headroom)

        generator = (
            torch.Generator(device="cpu").manual_seed(params.seed)
            if params.seed is not None
            else None
        )

        tokens: List[int] = list(prompt_ids)
        generated: List[int] = []
        sources: List[str] = []
        stats = SpecStats()

        block_table: List[int] = []
        target_cache: Optional[PagedCache] = None
        draft_cache = DynamicCache()
        target_len = 0  # tokens of `tokens` already in the target's paged cache
        draft_len = 0  # tokens of `tokens` already in the draft's cache

        start = time.perf_counter()
        first_token_at: Optional[float] = None
        finish_reason = "length"
        decoded_chars = 0

        try:
            while len(generated) < params.max_new_tokens:
                budget = params.max_new_tokens - len(generated)
                # The last proposal slot is covered by the bonus token, so never
                # propose more than we could still emit.
                k = max(1, min(self.lookahead_k, budget))

                # ---- 1. draft proposes k tokens -------------------------------
                drafts: List[int] = []
                draft_probs: List[torch.Tensor] = []
                ctx = list(tokens)
                for _ in range(k):
                    missing = ctx[draft_len:]
                    if missing:
                        ids = torch.tensor([missing], dtype=torch.long, device=self.device)
                        with torch.no_grad():
                            out = self.draft_model(
                                input_ids=ids, past_key_values=draft_cache, use_cache=True
                            )
                        stats.draft_forward_passes += 1
                        draft_len += len(missing)
                        logits = out.logits[0, -1]
                    else:  # pragma: no cover - defensive
                        break
                    q = logits_to_probs(logits, params.temperature, params.top_p)
                    d = sample_from_probs(q, generator)
                    drafts.append(d)
                    draft_probs.append(q)
                    ctx.append(d)
                    if params.stop_on_eos and d in self.eos_token_ids:
                        break

                if not drafts:  # pragma: no cover - defensive
                    break
                k_eff = len(drafts)
                stats.drafted += k_eff

                # ---- 2. target verifies all k in ONE forward pass -------------
                missing = tokens[target_len:]
                verify_input = missing + drafts
                self._grow_blocks(block_table, target_len + len(verify_input))

                if target_cache is None:
                    target_cache = PagedCache(self.pool, block_table, device=self.device)
                target_cache.begin_step(block_table)

                ids = torch.tensor([verify_input], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    out = self.target_model(
                        input_ids=ids, past_key_values=target_cache, use_cache=True
                    )
                stats.target_forward_passes += 1
                target_len += len(verify_input)

                # logits[base + i] is the target's distribution for the i-th
                # speculative position; base skips the already-known tokens.
                base = len(missing) - 1
                target_probs = [
                    logits_to_probs(out.logits[0, base + i], params.temperature, params.top_p)
                    for i in range(k_eff + 1)
                ]

                # ---- 3. rejection sampling -----------------------------------
                round_tokens: List[tuple] = []
                num_accepted = 0
                for i in range(k_eff):
                    d = drafts[i]
                    p_d = float(target_probs[i][d])
                    q_d = float(draft_probs[i][d])
                    threshold = min(1.0, p_d / q_d) if q_d > 0 else 0.0
                    r = float(torch.rand(1, generator=generator).item())
                    if r <= threshold:
                        round_tokens.append((d, "draft"))
                        num_accepted += 1
                    else:
                        residual = torch.clamp(target_probs[i] - draft_probs[i], min=0.0)
                        total = float(residual.sum())
                        corrected = sample_from_probs(
                            residual / total if total > 0 else target_probs[i], generator
                        )
                        round_tokens.append((corrected, "resample"))
                        stats.resampled += 1
                        break
                else:
                    # Every proposal survived: the verification pass also yields a
                    # free extra token from the target's own distribution.
                    bonus = sample_from_probs(target_probs[k_eff], generator)
                    round_tokens.append((bonus, "bonus"))
                    stats.bonus += 1

                stats.accepted += num_accepted
                stats.accepted_per_round.append(num_accepted)
                stats.rounds += 1

                # ---- 4. commit, and roll back the speculative tail -----------
                # The cache holds every proposed token; discard the ones that did
                # not survive. O(1): only the write cursor moves.
                rollback = k_eff - num_accepted
                if rollback > 0:
                    target_cache.rollback(rollback)
                    target_len -= rollback
                    stats.rollback_tokens += rollback

                # Keep the draft cache only up to the longest valid common prefix.
                valid_prefix = len(tokens) + num_accepted
                if draft_len > valid_prefix:
                    draft_cache.crop(-(draft_len - valid_prefix))
                    draft_len = valid_prefix

                for token_id, source in round_tokens:
                    if len(generated) >= params.max_new_tokens:
                        break
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    tokens.append(token_id)
                    generated.append(token_id)
                    sources.append(source)

                    full = self.tokenizer.decode(generated, skip_special_tokens=True)
                    delta = full[decoded_chars:]
                    decoded_chars = len(full)

                    yield SpecTokenEvent(
                        index=len(generated) - 1,
                        token_id=token_id,
                        text=delta,
                        source=source,
                        round_index=stats.rounds - 1,
                    )

                    if params.stop_on_eos and token_id in self.eos_token_ids:
                        finish_reason = "stop"
                        break

                if finish_reason == "stop":
                    break
                if len(tokens) >= self.max_model_len:
                    finish_reason = "context_limit"
                    break
        finally:
            if block_table:
                self.allocator.free_blocks(block_table)

        elapsed = time.perf_counter() - start
        ttft = (first_token_at - start) if first_token_at else None
        return {
            "text": self.tokenizer.decode(generated, skip_special_tokens=True),
            "token_ids": generated,
            "token_sources": sources,
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(generated),
            "finish_reason": finish_reason,
            "time_to_first_token_ms": round(ttft * 1000, 2) if ttft else None,
            "total_time_s": round(elapsed, 4),
            "tokens_per_second": round(len(generated) / elapsed, 2) if elapsed > 0 else 0.0,
            "lookahead_k": self.lookahead_k,
            "rounds": stats.rounds,
            "drafted_tokens": stats.drafted,
            "accepted_tokens": stats.accepted,
            "resampled_tokens": stats.resampled,
            "bonus_tokens": stats.bonus,
            "acceptance_rate": round(stats.acceptance_rate, 4),
            "mean_accepted_per_round": (
                round(sum(stats.accepted_per_round) / len(stats.accepted_per_round), 3)
                if stats.accepted_per_round
                else 0.0
            ),
            "target_forward_passes": stats.target_forward_passes,
            "draft_forward_passes": stats.draft_forward_passes,
            "tokens_per_target_pass": (
                round(len(generated) / stats.target_forward_passes, 3)
                if stats.target_forward_passes
                else 0.0
            ),
            "rolled_back_tokens": stats.rollback_tokens,
        }

    def generate(self, prompt, params: Optional[SamplingParams] = None) -> Dict[str, Any]:
        """Run speculative generation to completion and return the summary."""
        gen = self.stream(prompt, params)
        while True:
            try:
                next(gen)
            except StopIteration as stop:
                return stop.value
