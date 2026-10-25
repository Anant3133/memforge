"""
ContinuousBatchingEngine: the live inference runtime.

This is where the three MemForge subsystems finally meet on real weights:

  * the C++ `Scheduler` decides, every single token step, which sequences run --
    new requests join mid-flight and finished ones leave immediately, so there is
    no static batch to wait for (continuous batching);
  * the C++ `BlockAllocator` hands out 16-token physical blocks, and every
    sequence's KV cache lives in those blocks via `PagedCache` (PagedAttention);
  * generation itself is an ordinary Hugging Face forward pass, so the engine
    stays model-agnostic.

The decode path is batched: all sequences that need a token this step are gathered
out of their non-contiguous blocks into one left-padded batch, run through a single
forward pass, and scattered back into their own blocks.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from memforge_ml.runtime.kv_pool import ModelKVSpec, PagedKVPool
from memforge_ml.runtime.paged_cache import PagedCache
from memforge_ml.runtime.prefix_sharing import block_owners, copy_block, sharing_telemetry
from memforge_ml.runtime.sampling import SamplingParams, sample_logits

try:
    import memforge_c

    HAS_CPP_ENGINE = True
except ImportError:  # pragma: no cover - only on an unbuilt checkout
    memforge_c = None
    HAS_CPP_ENGINE = False


@dataclass
class TokenEvent:
    """A single emitted token, carrying what the UI needs to render it."""

    seq_id: int
    index: int
    token_id: int
    text: str
    source: str = "target"  # "target" | "draft" | "resample"


@dataclass
class SequenceRuntime:
    """Python-side state for one in-flight sequence (the C++ side owns memory)."""

    seq_id: int
    prompt_ids: List[int]
    params: SamplingParams
    tokens: List[int]
    cache: Optional[PagedCache] = None
    generated: List[int] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    events: Deque[TokenEvent] = field(default_factory=deque)
    generator: Optional[torch.Generator] = None
    submitted_at: float = field(default_factory=time.perf_counter)
    first_token_at: Optional[float] = None
    finished_at: Optional[float] = None
    finish_reason: Optional[str] = None
    num_prefills: int = 0
    num_preemptions: int = 0
    decoded_chars: int = 0

    @property
    def done(self) -> bool:
        return self.finish_reason is not None

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_ids)


@dataclass
class StepReport:
    """Telemetry for one iteration-level scheduler step."""

    step_index: int
    num_running: int
    num_waiting: int
    num_prefill: int
    num_decode: int
    tokens_emitted: int
    finished: List[int]
    preempted: List[int]
    aborted: List[int]
    free_blocks: int
    total_blocks: int
    memory_utilization: float
    step_latency_ms: float
    events: List[TokenEvent]


class ContinuousBatchingEngine:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        block_size: int = 16,
        num_blocks: Optional[int] = None,
        kv_budget_mb: float = 512.0,
        max_num_seqs: int = 16,
        max_model_len: int = 2048,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None,
        batched_decode: bool = True,
        enable_prefix_sharing: bool = True,
    ):
        if not HAS_CPP_ENGINE:
            raise RuntimeError(
                "The native memforge_c extension is required by ContinuousBatchingEngine. "
                "Build it with: python setup.py build_ext --inplace"
            )

        self.model = model
        self.tokenizer = tokenizer
        self.device = device or str(next(model.parameters()).device)
        self.dtype = dtype or next(model.parameters()).dtype
        self.block_size = block_size
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.batched_decode = batched_decode

        self.spec = ModelKVSpec.from_config(model.config, self.dtype, self.device)
        if num_blocks is None:
            num_blocks = PagedKVPool.blocks_for_budget(self.spec, block_size, kv_budget_mb)
        self.pool = PagedKVPool(self.spec, num_blocks=num_blocks, block_size=block_size)

        self.allocator = memforge_c.BlockAllocator(num_blocks, block_size)
        # Prefix sharing is opt-out so A/B benchmarking stays possible in one
        # process. With it disabled the scheduler behaves exactly as it did
        # before the feature existed.
        self.prefix_cache = (
            memforge_c.PrefixCache(self.allocator, block_size) if enable_prefix_sharing else None
        )
        self.scheduler = memforge_c.Scheduler(self.allocator, max_num_seqs, self.prefix_cache)

        self.eos_token_ids = self._resolve_eos_ids()
        self.runtimes: Dict[int, SequenceRuntime] = {}
        self._next_seq_id = 1
        self._step_index = 0

        # cumulative counters
        self.total_tokens_generated = 0
        self.total_prefill_tokens = 0
        self.total_steps = 0
        self.total_forward_passes = 0
        self.total_preemptions = 0
        self.total_cached_prefix_tokens = 0   # prefill tokens skipped via a hit
        self.total_cow_copies = 0
        self.engine_started_at = time.perf_counter()

        # Aborts arrive from whatever thread the server's event loop runs on,
        # while step() runs in a worker thread. They are queued here and applied
        # at the top of step(), so every mutation of engine state happens on the
        # stepping thread. See abort() for why that matters.
        self._abort_lock = threading.Lock()
        self._pending_aborts: set = set()
        # Timestamps of recently emitted tokens, for a windowed throughput figure.
        # The lifetime average counts idle time and so understates a live gauge.
        self.throughput_window_s = 5.0
        self._recent_tokens: Deque[float] = deque()

    # ------------------------------------------------------------------ setup

    def _resolve_eos_ids(self) -> set:
        ids = set()
        cfg_eos = getattr(self.model.config, "eos_token_id", None)
        for candidate in (cfg_eos, getattr(self.tokenizer, "eos_token_id", None)):
            if isinstance(candidate, int):
                ids.add(candidate)
            elif isinstance(candidate, (list, tuple)):
                ids.update(int(c) for c in candidate)
        return ids

    def describe(self) -> Dict[str, Any]:
        return {
            "engine": "C++ native (memforge_c)",
            "model": getattr(self.model.config, "name_or_path", "unknown"),
            "device": self.device,
            "dtype": str(self.dtype),
            "max_num_seqs": self.max_num_seqs,
            "max_model_len": self.max_model_len,
            "batched_decode": self.batched_decode,
            "prefix_sharing": self.prefix_cache is not None,
            "eos_token_ids": sorted(self.eos_token_ids),
            "kv_pool": self.pool.describe(),
        }

    # ------------------------------------------------------------- submission

    def submit(self, prompt, params: Optional[SamplingParams] = None) -> int:
        """Enqueue a request. Returns its sequence id; nothing runs until `step()`."""
        params = params or SamplingParams()
        if isinstance(prompt, str):
            prompt_ids = self.tokenizer(prompt, return_tensors=None)["input_ids"]
        else:
            prompt_ids = list(prompt)
        if not prompt_ids:
            raise ValueError("cannot submit an empty prompt")

        headroom = self.max_model_len - len(prompt_ids)
        if headroom <= 0:
            raise ValueError(
                f"prompt of {len(prompt_ids)} tokens leaves no room under "
                f"max_model_len={self.max_model_len}"
            )
        params = params.clamped(headroom)

        seq_id = self._next_seq_id
        self._next_seq_id += 1

        generator = None
        if params.seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(params.seed)

        self.runtimes[seq_id] = SequenceRuntime(
            seq_id=seq_id,
            prompt_ids=list(prompt_ids),
            params=params,
            tokens=list(prompt_ids),
            generator=generator,
        )
        self.scheduler.add_sequence(seq_id, list(prompt_ids), params.max_new_tokens)
        return seq_id

    def abort(self, seq_id: int) -> bool:
        """
        Request that a sequence be abandoned. Safe to call from any thread.

        The teardown is deferred to the next `step()` rather than done here, and
        that is not tidiness -- it fixes a real crash. The server aborts from its
        asyncio thread when a client disconnects, while `step()` runs in a worker
        thread. Tearing down inline raced with the step in progress two ways:
        clearing `rt.cache` made a live decode fail with
        `'NoneType' object has no attribute 'indices'`, and freeing the physical
        blocks mid-step was worse still, because the step would carry on writing
        into blocks the allocator had already handed to another sequence.

        Found by scripts/traffic_simulator.py --disconnect-rate: 32 of 96
        requests hit it under concurrent load with disconnects, and nothing else
        in the project exercised that combination.

        Returns True if the sequence existed and was still running.
        """
        rt = self.runtimes.get(seq_id)
        if rt is None or rt.done:
            return False
        with self._abort_lock:
            self._pending_aborts.add(seq_id)
        return True

    def _process_pending_aborts(self) -> None:
        """Apply queued aborts. Runs on the stepping thread only."""
        with self._abort_lock:
            if not self._pending_aborts:
                return
            pending = self._pending_aborts
            self._pending_aborts = set()

        for seq_id in pending:
            rt = self.runtimes.get(seq_id)
            if rt is None or rt.done:
                continue
            self.scheduler.abort_sequence(seq_id)
            self._finish(rt, "aborted")

    # ------------------------------------------------------------------ steps

    def has_work(self) -> bool:
        # A queued-but-unapplied abort counts as work, so the next step runs and
        # drains it rather than the engine going idle with it still pending.
        with self._abort_lock:
            if self._pending_aborts:
                return True
        return self.scheduler.has_unfinished_sequences() or any(
            not rt.done for rt in self.runtimes.values()
        )

    def step(self) -> StepReport:
        """Run exactly one iteration-level scheduling + execution step."""
        t0 = time.perf_counter()
        # Before anything reads engine state, apply aborts requested from other
        # threads, so the rest of this step sees a consistent view.
        self._process_pending_aborts()
        result = self.scheduler.step()
        self._step_index += 1
        self.total_steps += 1

        for seq_id in result.aborted_sequence_ids:
            rt = self.runtimes.get(seq_id)
            if rt is not None and not rt.done:
                self._finish(rt, "aborted_unschedulable")

        for seq_id in result.preempted_sequence_ids:
            rt = self.runtimes.get(seq_id)
            if rt is not None and rt.cache is not None:
                # Blocks were reclaimed by the scheduler; drop the stale mapping so
                # the sequence is recomputed from scratch on readmission.
                rt.cache = None
                rt.num_preemptions += 1
                self.total_preemptions += 1

        prefill_infos = [i for i in result.running_sequences if i.is_prefill]
        decode_infos = [i for i in result.running_sequences if not i.is_prefill]

        events: List[TokenEvent] = []
        for info in prefill_infos:
            ev = self._run_prefill(info)
            if ev is not None:
                events.append(ev)

        if decode_infos:
            if self.batched_decode and len(decode_infos) > 1:
                events.extend(self._run_decode_batched(decode_infos))
            else:
                events.extend(self._run_decode_sequential(decode_infos))

        for seq_id in result.finished_sequence_ids:
            rt = self.runtimes.get(seq_id)
            if rt is not None and not rt.done:
                self._finish(rt, "length")

        return StepReport(
            step_index=self._step_index,
            num_running=len(result.running_sequences),
            num_waiting=result.num_waiting,
            num_prefill=len(prefill_infos),
            num_decode=len(decode_infos),
            tokens_emitted=len(events),
            finished=list(result.finished_sequence_ids),
            preempted=list(result.preempted_sequence_ids),
            aborted=list(result.aborted_sequence_ids),
            free_blocks=result.num_free_blocks,
            total_blocks=result.total_blocks,
            memory_utilization=result.memory_utilization_ratio,
            step_latency_ms=(time.perf_counter() - t0) * 1000.0,
            events=events,
        )

    # ---------------------------------------------------------------- prefill

    def _run_prefill(self, info) -> Optional[TokenEvent]:
        rt = self.runtimes.get(info.seq_id)
        if rt is None or rt.done:
            return None

        block_table = list(info.block_table)
        if rt.cache is None:
            rt.cache = PagedCache(self.pool, block_table, device=self.device)
        else:
            rt.cache.reset()
        rt.cache.begin_step(block_table)
        rt.num_prefills += 1

        # Leading tokens whose KV a prefix-cache hit already put in the pool.
        # The scheduler guarantees this is strictly less than the sequence
        # length (it withholds the final full block), so the suffix below is
        # never empty -- a zero-length forward pass is undefined. The clamp is
        # belt-and-braces against that invariant ever being relaxed.
        cached = int(getattr(info, "num_cached_tokens", 0) or 0)
        if cached >= len(rt.tokens):
            cached = 0

        # On a fresh request this is just the prompt; after a preemption it is
        # prompt + everything generated so far, recomputed in one pass. Either
        # way, only the uncached suffix goes through the model.
        suffix = rt.tokens[cached:] if cached else rt.tokens
        ids = torch.tensor([suffix], dtype=torch.long, device=self.device)

        extra: Dict[str, Any] = {}
        if cached:
            # Advance the write cursor so the suffix lands *after* the cached
            # prefix instead of on top of it -- overwriting it would corrupt the
            # prefix for every other sequence sharing those blocks.
            rt.cache.seed_position(cached)
            # Positions must stay absolute, or RoPE rotates the suffix as though
            # the prefix were not there -- fluent output, quietly wrong.
            # `seed_position` alone is already enough for that, because
            # transformers derives the cache position from the cache's reported
            # length and PagedCache reports it correctly (a mutation test
            # confirms removing these two lines changes nothing). They stay
            # anyway: this is the one place where the engine knows the absolute
            # offset for certain, and stating it costs nothing and does not
            # depend on how a future transformers release infers it.
            positions = torch.arange(cached, len(rt.tokens), device=self.device)
            extra["position_ids"] = positions.unsqueeze(0)
            extra["cache_position"] = positions

        with torch.no_grad():
            out = self.model(input_ids=ids, past_key_values=rt.cache, use_cache=True, **extra)
        self.total_forward_passes += 1
        self.total_prefill_tokens += len(suffix)
        self.total_cached_prefix_tokens += cached

        # The whole sequence is resident now: the prefix was already there and
        # the suffix has just been written. Only now is it safe to advertise
        # these blocks for reuse -- registering before the KV exists would serve
        # the next sequence a block full of zeros.
        if self.prefix_cache is not None:
            self.prefix_cache.register_full_blocks(rt.tokens, block_table)

        token_id = sample_logits(out.logits[0, -1], rt.params, rt.generator)
        return self._accept_token(rt, token_id, source="target")

    # -------------------------------------------------------- copy-on-write

    def _begin_write_step(self, rt: SequenceRuntime, info) -> None:
        """
        Rebind a sequence's cache for this step, copying on write if needed.

        A write must never land in a block another sequence is reading. Today the
        allocator only ever shares *full* blocks and decode only ever writes the
        trailing block, which is private by construction -- so `prepare_write`
        returns None every time and this is a guard rather than a hot path. It
        exists because the alternative to an unreachable guard is an unprotected
        invariant: anything that later shares a non-final block (partial-block
        sharing, a scheduler change) would silently corrupt a live prefix.
        """
        block_table = list(info.block_table)
        rt.cache.begin_step(block_table)

        if self.prefix_cache is None:
            return
        seq = self.scheduler.get_sequence(rt.seq_id)
        if seq is None:
            return

        logical = rt.cache.position // self.block_size
        cow = self.allocator.prepare_write(seq, logical)
        if cow is None:
            return

        old_block, new_block = cow
        # The block table was repointed in C++; carry the old contents over
        # before the forward pass writes into the private copy, then rebind so
        # the cache's memoized logical->physical translation is rebuilt.
        copy_block(self.pool, old_block, new_block)
        rt.cache.set_block_table(list(seq.get_block_table()))
        self.total_cow_copies += 1

    # ----------------------------------------------------------------- decode

    def _run_decode_sequential(self, infos) -> List[TokenEvent]:
        events: List[TokenEvent] = []
        for info in infos:
            rt = self.runtimes.get(info.seq_id)
            if rt is None or rt.done or rt.cache is None:
                continue
            self._begin_write_step(rt, info)
            ids = torch.tensor([[rt.tokens[-1]]], dtype=torch.long, device=self.device)
            with torch.no_grad():
                out = self.model(input_ids=ids, past_key_values=rt.cache, use_cache=True)
            self.total_forward_passes += 1
            token_id = sample_logits(out.logits[0, -1], rt.params, rt.generator)
            ev = self._accept_token(rt, token_id, source="target")
            if ev is not None:
                events.append(ev)
        return events

    def _run_decode_batched(self, infos) -> List[TokenEvent]:
        """
        One forward pass for every decoding sequence.

        Each sequence's KV lives in its own scattered physical blocks, so the batch
        is assembled by gathering from the pool and left-padding to the longest
        context; `attention_mask` hides the padding and explicit `position_ids`
        keep RoPE aligned to each sequence's true position. After the pass, only
        the newly produced K/V slice is scattered back into each sequence's blocks.
        """
        rts: List[SequenceRuntime] = []
        for info in infos:
            rt = self.runtimes.get(info.seq_id)
            if rt is None or rt.done or rt.cache is None:
                continue
            self._begin_write_step(rt, info)
            rts.append(rt)

        if not rts:
            return []
        if len(rts) == 1:
            return self._run_decode_sequential(infos)

        ctx_lens = [rt.cache.position for rt in rts]
        max_len = max(ctx_lens)
        batch = len(rts)

        read_idx = [rt.cache.indices(0, n) for rt, n in zip(rts, ctx_lens)]

        layer_data = []
        for layer_idx in range(self.spec.num_layers):
            keys, values = [], []
            for (phys, slots), n in zip(read_idx, ctx_lens):
                k, v = self.pool.gather(layer_idx, phys, slots)
                pad = max_len - n
                if pad:
                    k = F.pad(k, (0, 0, pad, 0))
                    v = F.pad(v, (0, 0, pad, 0))
                keys.append(k)
                values.append(v)
            layer_data.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))

        # A throwaway contiguous cache seeded from the paged pool: the model sees a
        # normal batched cache, while the pool stays the source of truth.
        cache = DynamicCache(ddp_cache_data=layer_data)

        input_ids = torch.tensor(
            [[rt.tokens[-1]] for rt in rts], dtype=torch.long, device=self.device
        )
        attention_mask = torch.zeros((batch, max_len + 1), dtype=torch.long, device=self.device)
        for i, n in enumerate(ctx_lens):
            attention_mask[i, max_len - n :] = 1
        position_ids = torch.tensor([[n] for n in ctx_lens], dtype=torch.long, device=self.device)
        cache_position = torch.arange(max_len, max_len + 1, device=self.device)

        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                cache_position=cache_position,
                use_cache=True,
            )
        self.total_forward_passes += 1

        # Scatter the freshly computed K/V back into each sequence's own blocks.
        for layer_idx in range(self.spec.num_layers):
            new_k = cache.layers[layer_idx].keys[:, :, -1:, :]
            new_v = cache.layers[layer_idx].values[:, :, -1:, :]
            for i, (rt, n) in enumerate(zip(rts, ctx_lens)):
                phys, slots = rt.cache.indices(n, n + 1)
                self.pool.write(layer_idx, phys, slots, new_k[i : i + 1], new_v[i : i + 1])

        # Advance each paged cache's write cursor past the token just written.
        for rt in rts:
            for layer in rt.cache.layers:
                layer.position += 1

        events: List[TokenEvent] = []
        for i, rt in enumerate(rts):
            token_id = sample_logits(out.logits[i, -1], rt.params, rt.generator)
            ev = self._accept_token(rt, token_id, source="target")
            if ev is not None:
                events.append(ev)
        return events

    # ------------------------------------------------------------ bookkeeping

    def _accept_token(self, rt: SequenceRuntime, token_id: int, source: str) -> Optional[TokenEvent]:
        now = time.perf_counter()
        if rt.first_token_at is None:
            rt.first_token_at = now

        rt.tokens.append(token_id)
        rt.generated.append(token_id)
        rt.sources.append(source)
        self.total_tokens_generated += 1
        self._recent_tokens.append(now)

        is_eos = rt.params.stop_on_eos and token_id in self.eos_token_ids
        hit_limit = len(rt.generated) >= rt.params.max_new_tokens
        self.scheduler.update_sequence(rt.seq_id, token_id, bool(is_eos))

        text = self._decode_delta(rt)
        event = TokenEvent(
            seq_id=rt.seq_id,
            index=len(rt.generated) - 1,
            token_id=token_id,
            text=text,
            source=source,
        )
        rt.events.append(event)

        if is_eos:
            self._finish(rt, "stop")
        elif hit_limit:
            self._finish(rt, "length")
        return event

    def _decode_delta(self, rt: SequenceRuntime) -> str:
        """
        Incremental detokenization. Byte-level BPE pieces are not independently
        decodable, so decode the whole output and emit only the new suffix.
        """
        full = self.tokenizer.decode(rt.generated, skip_special_tokens=True)
        delta = full[rt.decoded_chars :]
        rt.decoded_chars = len(full)
        return delta

    def _finish(self, rt: SequenceRuntime, reason: str) -> None:
        if rt.done:
            return
        rt.finish_reason = reason
        rt.finished_at = time.perf_counter()
        rt.cache = None  # the scheduler frees the physical blocks on its next step

    # ----------------------------------------------------------------- output

    def result_for(self, seq_id: int) -> Dict[str, Any]:
        rt = self.runtimes[seq_id]
        elapsed = (rt.finished_at or time.perf_counter()) - rt.submitted_at
        ttft = (rt.first_token_at - rt.submitted_at) if rt.first_token_at else None
        return {
            "seq_id": seq_id,
            "prompt_tokens": rt.prompt_len,
            "generated_tokens": len(rt.generated),
            "text": self.tokenizer.decode(rt.generated, skip_special_tokens=True),
            "token_ids": list(rt.generated),
            "token_sources": list(rt.sources),
            "finish_reason": rt.finish_reason,
            "time_to_first_token_ms": round(ttft * 1000, 2) if ttft else None,
            "total_time_s": round(elapsed, 4),
            "tokens_per_second": round(len(rt.generated) / elapsed, 2) if elapsed > 0 else 0.0,
            "num_prefills": rt.num_prefills,
            "num_preemptions": rt.num_preemptions,
        }

    def drain_events(self, seq_id: int) -> List[TokenEvent]:
        rt = self.runtimes[seq_id]
        out = list(rt.events)
        rt.events.clear()
        return out

    def recent_tokens_per_second(self) -> float:
        """Throughput over the last `throughput_window_s` seconds of activity."""
        cutoff = time.perf_counter() - self.throughput_window_s
        while self._recent_tokens and self._recent_tokens[0] < cutoff:
            self._recent_tokens.popleft()
        if len(self._recent_tokens) < 2:
            return 0.0
        span = self._recent_tokens[-1] - self._recent_tokens[0]
        return round(len(self._recent_tokens) / span, 2) if span > 0 else 0.0

    def telemetry(self) -> Dict[str, Any]:
        uptime = time.perf_counter() - self.engine_started_at
        active = [rt for rt in self.runtimes.values() if not rt.done]
        free = self.allocator.get_num_free_blocks()
        total = self.allocator.get_total_blocks()
        sharing = sharing_telemetry(
            self.allocator,
            self.pool,
            self.prefix_cache,
            owners=self.block_owners(),
        )
        return {
            **sharing,
            "uptime_s": round(uptime, 2),
            "steps": self.total_steps,
            "forward_passes": self.total_forward_passes,
            "tokens_generated": self.total_tokens_generated,
            "prefill_tokens": self.total_prefill_tokens,
            "tokens_per_second": self.recent_tokens_per_second(),
            "lifetime_tokens_per_second": (
                round(self.total_tokens_generated / uptime, 2) if uptime > 0 else 0.0
            ),
            "tokens_per_forward_pass": (
                round(self.total_tokens_generated / self.total_forward_passes, 3)
                if self.total_forward_passes
                else 0.0
            ),
            "active_sequences": len(active),
            "waiting_sequences": self.scheduler.get_num_waiting(),
            "running_sequences": self.scheduler.get_num_running(),
            "preemptions": self.total_preemptions,
            "cached_prefix_tokens": self.total_cached_prefix_tokens,
            "cow_copies": self.total_cow_copies,
            "total_blocks": total,
            "free_blocks": free,
            "allocated_blocks": total - free,
            "memory_utilization": round(self.allocator.get_utilization_ratio(), 4),
            "block_size": self.block_size,
            "kv_pool_mb": self.pool.describe()["total_mb"],
        }

    def _live_block_tables(self) -> List[Any]:
        return [
            (seq_id, rt.cache.block_table)
            for seq_id, rt in self.runtimes.items()
            if not rt.done and rt.cache is not None
        ]

    def block_occupancy(self) -> List[int]:
        """
        Per-physical-block owner map for the UI heatmap: index -> sequence id,
        or 0 for a free block.

        A shared block has several owners and this reports only the last one
        seen. The shape is kept as-is because the dashboard's memory heatmap
        binds to it; `block_owners()` is the share-aware view alongside it.
        """
        occupancy = [0] * self.allocator.get_total_blocks()
        for seq_id, table in self._live_block_tables():
            for block_id in table:
                if 0 <= block_id < len(occupancy):
                    occupancy[block_id] = seq_id
        return occupancy

    def block_owners(self) -> Dict[str, List[int]]:
        """Block -> every sequence referencing it, so sharing is visible."""
        return block_owners(self._live_block_tables(), self.allocator.get_total_blocks())

    # ------------------------------------------------------- blocking helpers

    def run_until_idle(self, max_steps: int = 100_000) -> List[StepReport]:
        reports = []
        steps = 0
        while self.has_work() and steps < max_steps:
            reports.append(self.step())
            steps += 1
        return reports

    def generate(self, prompts, params: Optional[SamplingParams] = None) -> List[Dict[str, Any]]:
        """Submit one or many prompts and run the engine until all of them finish."""
        prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
        ids = [self.submit(p, params) for p in prompt_list]
        self.run_until_idle()
        return [self.result_for(i) for i in ids]
