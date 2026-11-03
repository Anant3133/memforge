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

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import torch
from transformers.cache_utils import DynamicCache

from memforge_ml.kernels import attention_backend
from memforge_ml.runtime.graphed_paged_decode import GraphedPagedDecoder
from memforge_ml.kernels.batched_decode import BatchedDecodeBuffers, assemble_batch
from memforge_ml.runtime.batched_speculative import (
    BatchedGraphedDraftPool,
    BatchedGraphedDraftState,
    DraftState,
    GraphedDraftPool,
    propose_draft_tokens,
    resolve_round,
    scatter_accepted_kv,
    verify_candidates_batched,
)
from memforge_ml.runtime.decode_view_cache import capacity_for
from memforge_ml.runtime.kv_pool import ModelKVSpec, PagedKVPool
from memforge_ml.runtime.paged_cache import PagedCache
from memforge_ml.runtime.prefix_sharing import block_owners, copy_block, sharing_telemetry
from memforge_ml.runtime.sampling import SamplingParams, sample_logits
from memforge_ml.runtime.varlen_cache import VarlenPackedCache

LOGGER = logging.getLogger(__name__)

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
    #: Bytes of decode-mirror budget this sequence holds, 0 if it has none.
    view_cache_bytes: int = 0
    #: Draft-model cache, created on this sequence's first speculative step.
    draft: Optional[DraftState] = None
    #: Draft tokens proposed for this sequence, and how many were accepted.
    num_draft_proposed: int = 0
    num_draft_accepted: int = 0

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
        decode_view_cache: bool = False,
        view_cache_budget_mb: float = 512.0,
        block_attention: bool = False,
        max_prefill_tokens_per_step: int = 0,
        draft_model=None,
        lookahead_k: int = 0,
        fused_prefill: bool = False,
        paged_attn_kernel: str = "gather",  # "gather" | "triton" (read KV from the pool, no copy)
        draft_cuda_graph=False,  # False | True (per-sequence graphs) | "batched"
        paged_cuda_graph=False,  # False | True | "auto" (on when CUDA + sdpa allow it)
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
        # Tier 2 is opt-in and budgeted: see _grant_view_capacity().
        if decode_view_cache and block_attention:
            raise ValueError(
                "decode_view_cache and block_attention are mutually exclusive: "
                "the mirror makes materialising the context cheap, block-wise "
                "attention avoids materialising it at all"
            )
        # Phase C bullet 2. A fused step writes K/V straight into the pool
        # through a packed cache, so a sequence holding a decode mirror would
        # be left with a stale one -- the mirror never sees the packed write.
        if fused_prefill and decode_view_cache:
            raise ValueError(
                "fused_prefill and decode_view_cache are mutually exclusive: a "
                "fused pass writes through a packed cache that the per-sequence "
                "mirror never sees, so the mirror would go stale"
            )
        self.fused_prefill = bool(fused_prefill)
        self.decode_view_cache = decode_view_cache
        self.view_cache_budget_bytes = int(max(0.0, view_cache_budget_mb) * 1024 * 1024)
        self._view_cache_bytes_in_use = 0
        self.total_view_cache_grants = 0
        self.total_view_cache_denials = 0

        self.spec = ModelKVSpec.from_config(model.config, self.dtype, self.device)
        if num_blocks is None:
            num_blocks = PagedKVPool.blocks_for_budget(self.spec, block_size, kv_budget_mb)
        self.pool = PagedKVPool(self.spec, num_blocks=num_blocks, block_size=block_size)

        # Reusable batch scratch for the decode path. Shape-cached, so a
        # context growing one token per step does not reallocate a buffer
        # pair every step -- which was most of the cost it removes.
        self.decode_buffers = BatchedDecodeBuffers(
            self.spec.num_layers, block_size=block_size
        )

        # Optional: the whole plain decode step (index translation, KV write,
        # gather, forward) as one replayed CUDA graph. See graphed_paged_decode.
        self._paged_graph = None
        graphs_ok = self._cuda_graphs_possible(model)
        if paged_cuda_graph == "auto":
            paged_cuda_graph = graphs_ok
        if draft_cuda_graph == "auto":
            draft_cuda_graph = (
                "batched" if graphs_ok and draft_model is not None and self._cuda_graphs_possible(draft_model) else False
            )
        if draft_cuda_graph is True:
            import warnings
            warnings.warn(
                "draft_cuda_graph=True (one graph per sequence) is superseded by 'batched', which stays "
                "flat as the batch grows (0.82-0.91x vs 0.43-0.78x at batch 1-8 on Qwen3-8B); "
                "pass draft_cuda_graph='batched' or 'auto'.",
                DeprecationWarning, stacklevel=2,
            )
        if paged_cuda_graph:
            self._paged_graph = GraphedPagedDecoder(
                model, self.pool, max_batch=max_num_seqs, max_model_len=max_model_len,
                attn_kernel=paged_attn_kernel,
            )

        self.allocator = memforge_c.BlockAllocator(num_blocks, block_size)
        # Prefix sharing is opt-out so A/B benchmarking stays possible in one
        # process. With it disabled the scheduler behaves exactly as it did
        # before the feature existed.
        self.prefix_cache = (
            memforge_c.PrefixCache(self.allocator, block_size) if enable_prefix_sharing else None
        )
        # Chunked prefill (Phase C). 0 means unlimited, i.e. a prompt is
        # prefilled in one step -- the original behaviour.
        self.max_prefill_tokens_per_step = max(0, int(max_prefill_tokens_per_step))

        # Speculative decoding (Phase D). Off unless a draft model and a
        # positive lookahead are both given, because speculation with no draft
        # is just decoding with extra bookkeeping.
        self.draft_model = draft_model
        self.lookahead_k = max(0, int(lookahead_k)) if draft_model is not None else 0
        # Optional: replay the draft's decode step as a CUDA graph. Opt-in
        # because it needs a CUDA device and ~max_model_len of static draft KV
        # per concurrent sequence.
        self._draft_pool = None
        if draft_cuda_graph and draft_model is not None and self.lookahead_k > 0:
            batched = draft_cuda_graph in ("batched", "batched-unrolled")
            extra = {"unroll": draft_cuda_graph == "batched-unrolled"} if batched else {}
            pool_cls = BatchedGraphedDraftPool if batched else GraphedDraftPool
            self._draft_pool = pool_cls(
                draft_model, max_len=self.max_model_len,
                num_slots=max_num_seqs, device=self.device, **extra,
            )
        # K draft tokens plus the bonus token all have to have somewhere to be
        # written before verification can reject any of them, so the scheduler
        # reserves K+1 slots per running sequence per step rather than 1.
        lookahead_slots = self.lookahead_k + 1 if self.lookahead_k > 0 else 0
        self.scheduler = memforge_c.Scheduler(
            self.allocator,
            max_num_seqs,
            self.prefix_cache,
            self.max_prefill_tokens_per_step,
            lookahead_slots,
        )

        # Tier 3 has to be installed on the model, not just enabled on the
        # engine, and registration can fail on a transformers without the
        # attention registry -- in which case the flag is reported as off
        # rather than silently doing nothing.
        # `register()`, not `enable_on()`: the model's attention is rerouted
        # only for the duration of a pass that has a binding. Rerouting it
        # permanently sends every other pass through the backend with nothing
        # bound, where it silently falls back to the backend's own eager
        # attention instead of the model's -- which is how block attention
        # came to run zero kernels under batched decode while reporting
        # itself enabled, and how it leaked into every engine sharing a model.
        self.block_attention = bool(block_attention) and attention_backend.register()
        if block_attention and not self.block_attention:
            LOGGER.warning(
                "block attention unavailable on this transformers version; "
                "falling back to the materialised read path"
            )
        self.total_blockwise_layers = 0
        self.total_blockwise_fallbacks = 0

        # A fused pass is only possible if the model actually routes attention
        # through this backend -- otherwise the varlen kernel is never called
        # and the packed tokens are attended as though they were one sequence.
        # Without the registry the feature cannot work at all, so it reports
        # itself off rather than producing wrong output.
        if self.fused_prefill and not attention_backend.register():
            LOGGER.warning(
                "fused prefill needs the attention registry, which this "
                "transformers version does not expose; running prefill and "
                "decode as separate passes"
            )
            self.fused_prefill = False

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
        # Prefill tokens this engine did not have to recompute. Counts BOTH
        # Phase B prefix-cache hits and Phase C chunk resumption, because the
        # scheduler reports them through the same `num_cached_tokens` field
        # and the engine cannot tell them apart. So this is a work-avoided
        # figure, not a sharing figure -- `prefix_cache_hits` is the one that
        # measures sharing. Conflating them once made a prefix-sharing test
        # pass with registration deleted.
        self.total_cached_prefix_tokens = 0
        self.total_cow_copies = 0
        self.total_prefill_chunks = 0
        self.total_fused_steps = 0
        self.total_fused_tokens = 0
        self.total_draft_tokens_proposed = 0
        self.total_draft_tokens_accepted = 0
        self.total_speculative_rounds = 0
        # Sum of batch sizes over rounds. Divided by the round count this is
        # the average number of sequences each single target pass verified --
        # the figure that says whether speculation and batching composed, or
        # merely coexisted.
        self.total_speculative_sequence_rounds = 0
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
            "decode_view_cache": self.decode_view_cache,
            "block_attention": self.block_attention,
            "max_prefill_tokens_per_step": self.max_prefill_tokens_per_step,
            "chunked_prefill": self.max_prefill_tokens_per_step > 0,
            "view_cache_budget_mb": round(
                self.view_cache_budget_bytes / (1024 * 1024), 2
            ),
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

    # ------------------------------------------------------- decode mirror

    def _view_capacity_bytes(self, capacity_tokens: int) -> int:
        """Bytes one sequence's mirror would occupy: K and V, every layer."""
        return (
            2
            * self.spec.num_layers
            * self.spec.num_kv_heads
            * capacity_tokens
            * self.spec.head_dim
            * torch.empty(0, dtype=self.dtype).element_size()
        )

    def _grant_view_capacity(self, rt: "SequenceRuntime") -> Optional[int]:
        """
        Decide whether this sequence gets a decode mirror, and how large.

        Sized from the sequence's own length rather than `max_model_len`, because
        the gap between them is usually most of the memory. Granted only while
        the shared budget allows: beyond it the sequence reads the pool as
        normal, which is slower but correct. That is what keeps enabling this a
        latency/memory trade rather than a way to run out of memory -- a global
        switch with no cap would make a burst of long requests fatal.
        """
        if not self.decode_view_cache:
            return None

        capacity = capacity_for(rt.prompt_len, rt.params.max_new_tokens)
        needed = self._view_capacity_bytes(capacity)
        if self._view_cache_bytes_in_use + needed > self.view_cache_budget_bytes:
            self.total_view_cache_denials += 1
            return None

        self._view_cache_bytes_in_use += needed
        self.total_view_cache_grants += 1
        rt.view_cache_bytes = needed
        return capacity

    def _release_view_capacity(self, rt: "SequenceRuntime") -> None:
        """Return a sequence's mirror budget. Idempotent."""
        if rt.view_cache_bytes:
            self._view_cache_bytes_in_use -= rt.view_cache_bytes
            # Clamp rather than trust the arithmetic: a negative in-use figure
            # would silently grant unlimited capacity from then on.
            self._view_cache_bytes_in_use = max(0, self._view_cache_bytes_in_use)
            rt.view_cache_bytes = 0

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
                # The mirror dies with the cache, so its budget must come
                # back or a preempted sequence would leak capacity and
                # starve every later one.
                self._release_view_capacity(rt)
                rt.cache = None
                rt.num_preemptions += 1
                self.total_preemptions += 1

        prefill_infos = [i for i in result.running_sequences if i.is_prefill]
        decode_infos = [i for i in result.running_sequences if not i.is_prefill]

        events: List[TokenEvent] = []

        # Phase C bullet 2: when a step carries both a prefill chunk and decode
        # tokens, run them as one packed pass instead of two passes through the
        # whole model. Two passes per step is precisely why the chunked-prefill
        # stall ratio floors at ~2.5x rather than approaching 1x.
        if self._can_fuse(prefill_infos, decode_infos):
            fused = self._run_fused_step(prefill_infos, decode_infos)
            if fused is not None:
                events.extend(fused)
                prefill_infos, decode_infos = [], []

        for info in prefill_infos:
            ev = self._run_prefill(info)
            if ev is not None:
                events.append(ev)

        if decode_infos:
            if self.speculation_enabled:
                events.extend(self._run_decode_speculative(decode_infos))
            elif self.batched_decode and len(decode_infos) > 1:
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
            rt.cache = PagedCache(
                self.pool,
                block_table,
                device=self.device,
                view_capacity=self._grant_view_capacity(rt),
                lazy_read=self.block_attention,
            )
        else:
            rt.cache.reset()
        rt.cache.begin_step(block_table)
        rt.num_prefills += 1

        # Where this step's prefill starts: tokens already resident, either from
        # a prefix-cache hit (Phase B) or from an earlier chunk of this
        # sequence's own prefill (Phase C).
        cached = int(getattr(info, "num_cached_tokens", 0) or 0)
        total = len(rt.tokens)

        # Where it ends. The scheduler caps this when chunked prefill is on, so
        # a long prompt is spread over steps instead of monopolising one.
        chunk_end = int(getattr(info, "prefill_chunk_end", 0) or 0)
        if chunk_end <= 0 or chunk_end > total:
            # Older scheduler, or an unchunked step: do the whole remainder.
            chunk_end = total
        if cached >= total:
            # Nothing resident can exceed the sequence; recompute from scratch
            # rather than attempt a zero-length forward pass, which is undefined.
            cached, chunk_end = 0, total

        if chunk_end <= cached:
            # The prefill budget ran out this step. The sequence stays running
            # and resumes next step; emitting nothing now is the point of
            # chunking, since it is what leaves the budget for decodes.
            rt.num_prefills -= 1
            return None

        is_final_chunk = chunk_end >= total

        # On a fresh request this is the prompt; after a preemption it is prompt
        # plus everything generated so far. Either way only this chunk goes
        # through the model.
        suffix = rt.tokens[cached:chunk_end]
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
            positions = torch.arange(cached, chunk_end, device=self.device)
            extra["position_ids"] = positions.unsqueeze(0)
            extra["cache_position"] = positions

        with torch.no_grad():
            out = self.model(input_ids=ids, past_key_values=rt.cache, use_cache=True, **extra)
        self.total_forward_passes += 1
        self.total_prefill_tokens += len(suffix)
        self.total_cached_prefix_tokens += cached

        if not is_final_chunk:
            # Mid-prefill: the logits here are for a token in the middle of the
            # prompt, not a continuation, so sampling from them would emit
            # garbage. The sequence produces its first token only once the whole
            # prompt is resident.
            self.total_prefill_chunks += 1
            return None

        # The whole sequence is resident now. Only at this point is it safe to
        # advertise these blocks for reuse -- registering a partially prefilled
        # sequence would serve the next one blocks whose KV is still zeros.
        if self.prefix_cache is not None:
            self.prefix_cache.register_full_blocks(rt.tokens, block_table)

        self.total_prefill_chunks += 1
        token_id = sample_logits(out.logits[0, -1], rt.params, rt.generator)
        return self._accept_token(rt, token_id, source="target")

    def _run_decode_blockwise(self, rts: List[SequenceRuntime]) -> List[TokenEvent]:
        """
        Batched decode with attention computed block by block out of the pool.

        One packed pass over one query token per sequence, using the same varlen
        kernel Phase C built for prefill/decode fusion -- in its block-wise
        form, so working memory stays `O(block_size)` instead of growing with
        the longest context in the batch. That bound is the whole point of the
        feature; the materialised variant is faster and would defeat it.

        Expect this to be *slower* than the ordinary batched path on CPU, for
        the reason documented in `block_attention.py`: iterating blocks in
        Python costs more than one vectorised gather. It is the shape that pays
        off where the per-block work can hide the overhead.
        """
        block_tables = [
            PagedKVPool.block_table_tensor(list(rt.cache.block_table), self.device)
            for rt in rts
        ]
        seq_ids = list(range(len(rts)))
        positions = [rt.cache.position for rt in rts]
        token_ids = [rt.tokens[-1] for rt in rts]

        cache = VarlenPackedCache(
            self.pool, self.spec.num_layers, block_tables, seq_ids, positions
        )
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        position_ids = torch.tensor([positions], dtype=torch.long, device=self.device)

        # Scoped, not permanent: leaving the model rerouted would send every
        # other pass through the backend with nothing bound, which is precisely
        # the defect this method was written to fix.
        with attention_backend.backend_enabled_on(self.model) as routed:
            if not routed:
                raise RuntimeError(
                    "block attention needs the attention registry, which is "
                    "unavailable; it should have been reported as off"
                )
            with attention_backend.varlen_attention_context(
                self.pool, block_tables, seq_ids, positions, block_wise=True
            ) as bound:
                with torch.no_grad():
                    out = self.model(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        past_key_values=cache,
                        use_cache=True,
                    )
        if bound.used != self.spec.num_layers:
            raise RuntimeError(
                f"block-wise decode: {bound.used} of {self.spec.num_layers} "
                f"layers used the kernel"
            )
        self.total_forward_passes += 1
        self.total_blockwise_layers += bound.used

        events: List[TokenEvent] = []
        for row, rt in enumerate(rts):
            for layer in rt.cache.layers:
                layer.position += 1
            token_id = sample_logits(out.logits[0, row], rt.params, rt.generator)
            ev = self._accept_token(rt, token_id, source="target")
            if ev is not None:
                events.append(ev)
        return events

    # ------------------------------------------------------------ fused step

    def _can_fuse(self, prefill_infos, decode_infos) -> bool:
        """
        Whether this step is a candidate for one packed pass.

        Both kinds of work have to be present -- fusing a step that has only
        one kind would just be the ordinary path with extra bookkeeping -- and
        speculation is excluded because its decode step is already a batched
        multi-query pass of its own with a different shape.

        The prefill/decode presence test here is an early exit, not a
        correctness guard: `_run_fused_step` checks the same thing again after
        preparation, because the prefill budget can leave a chunk empty and
        turn a mixed step into a decode-only one. Removing this check changes
        no output, only the wasted preparation work -- which mutation testing
        confirms. It is kept because that preparation resets and rebinds every
        participating cache, and doing it twice in a step is pure cost.
        """
        return (
            self.fused_prefill
            and not self.speculation_enabled
            and bool(prefill_infos)
            and bool(decode_infos)
        )

    def _prepare_fused_prefill(self, info, slot: int):
        """
        Set a prefilling sequence up for a packed pass.

        Mirrors `_run_prefill`'s chunk arithmetic exactly -- the two must agree
        on where a chunk starts and ends, or a fused step and an unfused step
        would write different KV for the same prompt. Returns None when this
        sequence contributes nothing this step.
        """
        rt = self.runtimes.get(info.seq_id)
        if rt is None or rt.done:
            return None

        block_table = list(info.block_table)
        if rt.cache is None:
            rt.cache = PagedCache(
                self.pool,
                block_table,
                device=self.device,
                # No mirror and no lazy flag: a fused pass reads the pool
                # through the varlen kernel and never asks this cache for a
                # context, so neither setting would have any effect here.
                lazy_read=False,
            )
        else:
            rt.cache.reset()
        rt.cache.begin_step(block_table)

        cached = int(getattr(info, "num_cached_tokens", 0) or 0)
        total = len(rt.tokens)
        chunk_end = int(getattr(info, "prefill_chunk_end", 0) or 0)
        if chunk_end <= 0 or chunk_end > total:
            chunk_end = total
        if cached >= total:
            cached, chunk_end = 0, total
        if chunk_end <= cached:
            # Budget exhausted for this sequence; it resumes next step.
            return None

        rt.num_prefills += 1
        return {
            "rt": rt,
            "kind": "prefill",
            "slot": slot,
            "block_table": list(block_table),
            "tokens": list(rt.tokens[cached:chunk_end]),
            "positions": list(range(cached, chunk_end)),
            "cached": cached,
            "chunk_end": chunk_end,
            "is_final": chunk_end >= total,
        }

    def _prepare_fused_decode(self, info, slot: int):
        rt = self.runtimes.get(info.seq_id)
        if rt is None or rt.done or rt.cache is None:
            return None
        self._begin_write_step(rt, info)
        return {
            "rt": rt,
            "kind": "decode",
            "slot": slot,
            "block_table": list(info.block_table),
            "tokens": [rt.tokens[-1]],
            "positions": [rt.cache.position],
        }

    def _run_fused_step(self, prefill_infos, decode_infos):
        """
        One forward pass serving a prefill chunk and several decode tokens.

        The tokens are packed into a single row with no padding, and each one
        carries the sequence it belongs to and its absolute position. Attention
        masks from those positions rather than from a batch mask, which is the
        only way to express a pass where every row has a different context
        length; see `kernels/varlen_attention.py`.

        Returns None if the step turns out not to be fusible after preparation
        -- for instance when the prefill budget left the chunk empty -- and the
        caller then runs the ordinary unfused paths. Preparation is side-effecting
        on the caches, so this must not be called speculatively and then
        discarded; the "not fusible" cases are all ones where nothing was
        prepared.
        """
        plan = []
        for info in prefill_infos:
            entry = self._prepare_fused_prefill(info, len(plan))
            if entry is not None:
                plan.append(entry)
        prefill_count = len(plan)
        for info in decode_infos:
            entry = self._prepare_fused_decode(info, len(plan))
            if entry is not None:
                plan.append(entry)

        # Still needs both kinds, or there is nothing to fuse and the ordinary
        # paths do it better. Nothing has been written yet at this point, so
        # handing the step back is safe.
        if prefill_count == 0 or len(plan) == prefill_count:
            for entry in plan:
                if entry["kind"] == "prefill":
                    entry["rt"].num_prefills -= 1
            return None

        block_tables = [
            PagedKVPool.block_table_tensor(entry["block_table"], self.device)
            for entry in plan
        ]
        seq_ids: List[int] = []
        positions: List[int] = []
        token_ids: List[int] = []
        for entry in plan:
            entry["rows"] = list(range(len(seq_ids), len(seq_ids) + len(entry["tokens"])))
            seq_ids.extend([entry["slot"]] * len(entry["tokens"]))
            positions.extend(entry["positions"])
            token_ids.extend(entry["tokens"])

        cache = VarlenPackedCache(
            self.pool, self.spec.num_layers, block_tables, seq_ids, positions
        )
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        # Absolute positions, so RoPE sees each token where it really sits. A
        # packed run's index means nothing across sequences, so this cannot be
        # left to transformers to infer.
        position_ids = torch.tensor([positions], dtype=torch.long, device=self.device)

        # The backend is switched on for this forward only. Leaving the model
        # rerouted would send ordinary decodes through it as well, where they
        # would find no binding and silently fall back to eager attention
        # instead of the model's configured implementation.
        with attention_backend.backend_enabled_on(self.model) as routed:
            if not routed:
                raise RuntimeError(
                    "fused pass needs the attention registry, which is "
                    "unavailable; fused_prefill should have been disabled"
                )
            with attention_backend.varlen_attention_context(
                self.pool, block_tables, seq_ids, positions
            ) as bound:
                with torch.no_grad():
                    out = self.model(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        past_key_values=cache,
                        use_cache=True,
                    )
        if bound.used != self.spec.num_layers:
            # Every layer must have taken the varlen path. A layer that fell
            # back would have used transformers' mask, which for a packed run
            # lets one sequence attend to another -- fluent, confident, wrong.
            raise RuntimeError(
                f"fused pass: {bound.used} of {self.spec.num_layers} layers used "
                f"the varlen kernel"
            )

        self.total_forward_passes += 1
        self.total_fused_steps += 1
        self.total_fused_tokens += len(token_ids)

        events: List[TokenEvent] = []
        for entry in plan:
            rt = entry["rt"]
            last_row = entry["rows"][-1]
            if entry["kind"] == "prefill":
                # The pass wrote this chunk's K/V through the packed cache, so
                # the sequence's own cursor is moved here rather than by its
                # cache's update().
                rt.cache.seed_position(entry["chunk_end"])
                self.total_prefill_tokens += len(entry["tokens"])
                self.total_cached_prefix_tokens += entry["cached"]
                self.total_prefill_chunks += 1
                if not entry["is_final"]:
                    # Mid-prompt logits are not a continuation; sampling them
                    # would emit garbage. The first token comes when the whole
                    # prompt is resident.
                    continue
                if self.prefix_cache is not None:
                    self.prefix_cache.register_full_blocks(
                        rt.tokens, entry["block_table"]
                    )
            else:
                for layer in rt.cache.layers:
                    layer.position += 1

            token_id = sample_logits(out.logits[0, last_row], rt.params, rt.generator)
            ev = self._accept_token(rt, token_id, source="target")
            if ev is not None:
                events.append(ev)

        return events

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

    def _decode_prepared(self, infos) -> List[TokenEvent]:
        """
        Ordinary decode for sequences whose `_begin_write_step` has already run.

        Only the speculative path needs this: it decides per sequence whether to
        speculate, and the ones that cannot must not have their cache rebound a
        second time in the same step.
        """
        if self.batched_decode and len(infos) > 1:
            return self._run_decode_batched(infos, prepared=True)
        return self._run_decode_sequential(infos, prepared=True)

    def _run_decode_sequential(self, infos, prepared: bool = False) -> List[TokenEvent]:
        events: List[TokenEvent] = []
        for info in infos:
            rt = self.runtimes.get(info.seq_id)
            if rt is None or rt.done or rt.cache is None:
                continue
            if not prepared:
                self._begin_write_step(rt, info)
            if self._paged_graph is not None and not self.block_attention:
                events.extend(self._decode_graphed([rt]))
                continue
            ids = torch.tensor([[rt.tokens[-1]]], dtype=torch.long, device=self.device)
            if self.block_attention:
                # The backend has no reference to the cache from its own
                # signature, so the pool and this sequence's cache are bound for
                # the duration of the pass. Counters let telemetry show whether
                # the kernel was actually taken -- a silent fallback would look
                # exactly like success.
                with attention_backend.backend_enabled_on(self.model):
                    with attention_backend.paged_attention_context(
                        self.pool, rt.cache
                    ) as bound:
                        with torch.no_grad():
                            out = self.model(
                                input_ids=ids, past_key_values=rt.cache, use_cache=True
                            )
                self.total_blockwise_layers += bound.used
                self.total_blockwise_fallbacks += bound.fallbacks
            else:
                with torch.no_grad():
                    out = self.model(
                        input_ids=ids, past_key_values=rt.cache, use_cache=True
                    )
            self.total_forward_passes += 1
            token_id = sample_logits(out.logits[0, -1], rt.params, rt.generator)
            ev = self._accept_token(rt, token_id, source="target")
            if ev is not None:
                events.append(ev)
        return events

    def _run_decode_batched(self, infos, prepared: bool = False) -> List[TokenEvent]:
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
            if not prepared:
                self._begin_write_step(rt, info)
            rts.append(rt)

        if not rts:
            return []
        if self._paged_graph is not None and not self.block_attention:
            return self._decode_graphed(rts)
        if len(rts) == 1:
            # The write step already ran above, so it must not run again.
            return self._run_decode_sequential(infos, prepared=True)

        if self.block_attention:
            # Block-wise attention cannot be served by the left-padded gather
            # below: that gather materialises every sequence's whole context,
            # which is the exact cost this feature exists to avoid. A batched
            # decode is a packed varlen pass with one query token per sequence,
            # so it goes through that path instead.
            #
            # This branch is what makes the feature real under the default
            # configuration. Before it existed the batched path opened no
            # binding at all, so every layer fell through to the backend's
            # eager attention -- the kernel ran zero times while telemetry
            # reported the feature on, and the output quietly differed from
            # the baseline.
            return self._run_decode_blockwise(rts)

        ctx_lens = [rt.cache.position for rt in rts]
        batch = len(rts)

        # Gather every sequence directly into one preallocated, left-padded
        # buffer. The previous path copied each sequence three times per layer
        # -- gather, F.pad, torch.cat -- which measured 46.8 ms and 640 MB per
        # step at batch 8 / context 2048. Here padding is the region nobody
        # wrote and concatenation never happens.
        block_tables = [rt.cache.block_table_tensor_cached() for rt in rts]
        layer_data, batch_mask, padded_len = assemble_batch(
            self.pool, self.decode_buffers, block_tables, ctx_lens
        )

        # The model sees an ordinary batched cache; the pool stays the source of
        # truth and the new K/V is scattered back into it after the pass.
        cache = DynamicCache(ddp_cache_data=layer_data)

        input_ids = torch.tensor(
            [[rt.tokens[-1]] for rt in rts], dtype=torch.long, device=self.device
        )
        # One extra column for the token being generated this step.
        attention_mask = torch.zeros(
            (batch, padded_len + 1), dtype=torch.long, device=self.device
        )
        attention_mask[:, :padded_len] = batch_mask
        attention_mask[:, padded_len] = 1
        position_ids = torch.tensor([[n] for n in ctx_lens], dtype=torch.long, device=self.device)
        cache_position = torch.arange(padded_len, padded_len + 1, device=self.device)

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

    def _cuda_graphs_possible(self, model) -> bool:
        """CUDA graphs need a CUDA device and the sdpa attention backend (see GraphedPagedDecoder)."""
        return bool(
            str(self.device).startswith("cuda") and torch.cuda.is_available()
            and getattr(model.config, "_attn_implementation", None) == "sdpa"
        )

    def _decode_graphed(self, rts: List[SequenceRuntime]) -> List[TokenEvent]:
        """Plain decode for `rts` as one graph replay (writes KV in-graph)."""
        ctx = [rt.cache.position for rt in rts]
        logits = self._paged_graph.step(
            [rt.tokens[-1] for rt in rts], ctx,
            [rt.cache.block_table_tensor_cached() for rt in rts],
        )
        self.total_forward_passes += 1
        events: List[TokenEvent] = []
        for i, rt in enumerate(rts):
            for layer in rt.cache.layers:
                layer.position += 1
            ev = self._accept_token(
                rt, sample_logits(logits[i], rt.params, rt.generator), source="target"
            )
            if ev is not None:
                events.append(ev)
        return events

    # ----------------------------------------------------- speculative decode

    @property
    def speculation_enabled(self) -> bool:
        return self.draft_model is not None and self.lookahead_k > 0

    def _speculation_headroom(self, rt: SequenceRuntime) -> bool:
        """
        Whether this sequence can safely take K+1 more tokens this step.

        A sequence near `max_model_len` or near its own `max_new_tokens` would
        need a shorter candidate run, and the batched verification requires a
        uniform width. Rather than pad rows to different real lengths -- which
        is the varlen problem -- such a sequence sits out this step and decodes
        one token the ordinary way.
        """
        span = self.lookahead_k + 1
        if rt.cache.position + span > self.max_model_len:
            return False
        return len(rt.generated) + span <= rt.params.max_new_tokens

    def _run_decode_speculative(self, infos) -> List[TokenEvent]:
        """
        One speculative round across every decoding sequence.

        The draft runs per sequence (K cheap passes each), then a *single*
        target pass verifies all of their candidate runs at once. That batched
        verification is what makes speculation and continuous batching compose:
        the naive composition costs one target pass per sequence per round,
        which is strictly worse than not speculating at all.

        Sequences without room for a full K+1 run fall back to ordinary decode
        in the same step, so one nearly-finished sequence does not disable
        speculation for the rest of the batch.
        """
        candidates_rts: List[SequenceRuntime] = []
        plain_infos = []
        for info in infos:
            rt = self.runtimes.get(info.seq_id)
            if rt is None or rt.done or rt.cache is None:
                continue
            self._begin_write_step(rt, info)
            if self._speculation_headroom(rt):
                candidates_rts.append(rt)
            else:
                plain_infos.append(info)

        events: List[TokenEvent] = []
        # `_begin_write_step` has already run for these, and running it twice in
        # a step is not idempotent, so the fallback path is given the infos
        # directly rather than re-entering the normal decode entry points.
        if plain_infos:
            events.extend(self._decode_prepared(plain_infos))
        if not candidates_rts:
            return events

        k = self.lookahead_k
        # ---------------------------------------------------- 1. propose
        proposals: List[List[int]] = []
        draft_probs: List[List[torch.Tensor]] = []
        token_counts: List[int] = []
        for rt in candidates_rts:
            if rt.draft is None:
                rt.draft = (
                    self._draft_pool.acquire() if self._draft_pool is not None else DraftState()
                )
            token_counts.append(len(rt.tokens))
        temps = {rt.params.temperature for rt in candidates_rts}
        if isinstance(self._draft_pool, BatchedGraphedDraftPool) and len(temps) == 1:
            # All sequences drafted in K+1 replays total, not per sequence.
            out = self._draft_pool.propose(
                [rt.draft for rt in candidates_rts],
                [rt.tokens for rt in candidates_rts], k, temps.pop(),
            )
            for cand, probs in out:
                proposals.append(cand)
                draft_probs.append(probs)
        else:
            for rt in candidates_rts:
                if isinstance(rt.draft, BatchedGraphedDraftState):
                    raise RuntimeError("batched draft needs one temperature per round")
                cand, probs = propose_draft_tokens(
                    self.draft_model, rt.draft, rt.tokens, k,
                    rt.params.temperature, self.device,
                )
                proposals.append(cand)
                draft_probs.append(probs)

        # ---------------------------------------------- 2. verify, batched
        ctx_lens = [rt.cache.position for rt in candidates_rts]
        block_tables = [rt.cache.block_table_tensor_cached() for rt in candidates_rts]
        queries = [
            [rt.tokens[-1]] + cand
            for rt, cand in zip(candidates_rts, proposals)
        ]
        logits, new_kv, _ = verify_candidates_batched(
            self.model, self.pool, self.decode_buffers,
            block_tables, ctx_lens, queries, self.device,
        )
        self.total_forward_passes += 1
        self.total_speculative_rounds += 1
        self.total_speculative_sequence_rounds += len(candidates_rts)

        # ------------------------------------------------------ 3. accept
        for row, rt in enumerate(candidates_rts):
            round_ = resolve_round(
                rt.seq_id, proposals[row], draft_probs[row], logits[row],
                rt.params.temperature,
            )

            # Emit first, so a sequence that stops on EOS part-way through the
            # run does not get the rest of the run written into its cache.
            confirmed = 0
            for position, token_id in enumerate(round_.accepted):
                source = "draft" if round_.is_draft[position] else "target"
                ev = self._accept_token(rt, token_id, source=source)
                confirmed += 1
                if ev is not None:
                    events.append(ev)
                if rt.done:
                    break

            rt.num_draft_proposed += len(round_.candidates)
            rt.num_draft_accepted += round_.num_accepted_draft
            self.total_draft_tokens_proposed += len(round_.candidates)
            self.total_draft_tokens_accepted += round_.num_accepted_draft

            if rt.done:
                # `_finish` already dropped the cache and the scheduler will
                # free these blocks on its next step. Writing now would land in
                # blocks about to belong to another sequence, and nothing will
                # ever read this sequence's KV again -- so there is nothing to
                # write and nothing to roll back.
                continue

            scatter_accepted_kv(
                self.pool, block_tables[row], ctx_lens[row], new_kv, row, confirmed
            )
            for layer in rt.cache.layers:
                layer.position += confirmed

            # Discard only the rejected tail, keeping the accepted candidates'
            # draft KV. This does not change the *number* of draft passes --
            # the next round's catch-up is one pass either way -- only its
            # width, since the catch-up then covers the newly confirmed tokens
            # instead of every accepted candidate as well.
            rt.draft.rollback_to(token_counts[row] + round_.num_accepted_draft)

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
        self._release_view_capacity(rt)
        if self._draft_pool is not None and rt.draft is not None:
            self._draft_pool.release(rt.draft)
            rt.draft = None
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

    def _device_telemetry(self) -> Dict[str, Any]:
        """
        Where the KV pool lives, and what the device actually has allocated.

        `peak_memory_mb` comes from `torch.cuda.max_memory_allocated`, which has
        no CPU equivalent -- there is no allocator to ask -- so on CPU it is
        reported as None rather than 0.0. Zero would read as "measured, and it
        was nothing", which is exactly the kind of number that makes a dashboard
        lie. This was Phase E's first bullet and could not be filled until a
        CUDA build was installed.

        `pool_mb` is what MemForge itself reserved up front; `allocated_mb` is
        everything torch holds, weights included. The gap between them is the
        model, so both are worth showing.
        """
        pool = self.pool.describe()
        payload: Dict[str, Any] = {
            "device": pool["device"],
            "dtype": pool["dtype"],
            "num_blocks": pool["num_blocks"],
            "block_size": pool["block_size"],
            "capacity_tokens": pool["capacity_tokens"],
            "pool_mb": pool["total_mb"],
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available() and str(pool["device"]).startswith("cuda"):
            mb = 1024 * 1024
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            payload.update(
                {
                    "device_name": torch.cuda.get_device_name(0),
                    "allocated_mb": round(torch.cuda.memory_allocated() / mb, 2),
                    "peak_memory_mb": round(torch.cuda.max_memory_allocated() / mb, 2),
                    "reserved_mb": round(torch.cuda.memory_reserved() / mb, 2),
                    "vram_total_mb": round(total_bytes / mb, 2),
                    "vram_free_mb": round(free_bytes / mb, 2),
                }
            )
        else:
            # None, not 0.0: nothing measured it, which is different from
            # measuring it and finding nothing.
            payload.update(
                {
                    "device_name": None,
                    "allocated_mb": None,
                    "peak_memory_mb": None,
                    "reserved_mb": None,
                    "vram_total_mb": None,
                    "vram_free_mb": None,
                }
            )
        return payload

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
            **self._device_telemetry(),
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
            "prefill_chunks": self.total_prefill_chunks,
            "fused_prefill_enabled": self.fused_prefill,
            "fused_steps": self.total_fused_steps,
            "fused_tokens": self.total_fused_tokens,
            "chunked_prefill_enabled": self.max_prefill_tokens_per_step > 0,
            "max_prefill_tokens_per_step": self.max_prefill_tokens_per_step,
            "speculation_enabled": self.speculation_enabled,
            "lookahead_k": self.lookahead_k,
            "speculative_rounds": self.total_speculative_rounds,
            "speculative_sequences_per_round": (
                round(
                    self.total_speculative_sequence_rounds
                    / self.total_speculative_rounds,
                    3,
                )
                if self.total_speculative_rounds
                else 0.0
            ),
            "draft_tokens_proposed": self.total_draft_tokens_proposed,
            "draft_tokens_accepted": self.total_draft_tokens_accepted,
            # The figure that decides whether speculation is worth its cost: at
            # a low acceptance rate the draft's passes are pure overhead, which
            # is exactly what the CPU measurements in the plan show.
            "draft_acceptance_rate": (
                round(
                    self.total_draft_tokens_accepted
                    / self.total_draft_tokens_proposed,
                    4,
                )
                if self.total_draft_tokens_proposed
                else 0.0
            ),
            "decode_view_cache_enabled": self.decode_view_cache,
            "block_attention_enabled": self.block_attention,
            "blockwise_layer_calls": self.total_blockwise_layers,
            "blockwise_fallbacks": self.total_blockwise_fallbacks,
            "view_cache_mb_in_use": round(
                self._view_cache_bytes_in_use / (1024 * 1024), 3
            ),
            "view_cache_budget_mb": round(
                self.view_cache_budget_bytes / (1024 * 1024), 2
            ),
            "view_cache_grants": self.total_view_cache_grants,
            "view_cache_denials": self.total_view_cache_denials,
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
