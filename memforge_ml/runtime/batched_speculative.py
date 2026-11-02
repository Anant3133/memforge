"""
Speculative decoding inside continuous batching (Phase D).

The existing `PagedSpeculativeEngine` is single-sequence and serialised behind a
lock, so speculation and batching could not be used together — limitation 4 in
`PROGRESS_AND_NEXT_PHASES.md` §7. This is the piece that composes them.

The shape of one speculative step, for N running sequences with lookahead K:

  1. **Propose.** Each sequence's draft model produces K candidate tokens, one
     forward pass at a time. This stays per-sequence: the draft is the cheap
     model, and batching it would mean a second set of padded contexts for a
     saving that the measured numbers do not justify.
  2. **Verify, batched.** *One* target forward pass covers every sequence's
     whole candidate run — `(N, K+1)` query positions against N paged contexts.
     This is Phase D's second bullet and the reason the phase exists: without it
     a batch of N sequences needs N target passes per speculative round, which
     is strictly worse than not speculating.
  3. **Accept.** Per sequence, standard rejection sampling against the target's
     own distribution, so the output is the target's distribution exactly. A
     rejected candidate truncates that sequence's run; the others are unaffected.

**Why K+1 and not K.** The target pass is fed
`[last_confirmed_token, d1, ..., dK]`, and its output at position *j* is the
distribution for the token after position *j*. So position 0 judges `d1`,
position *i* judges `d(i+1)`, and position K yields a free "bonus" token that
needs no draft at all. That is why a round can confirm K+1 tokens from one
target pass, and why the scheduler reserves K+1 slots
(`Scheduler(lookahead_slots=K+1)`) — a rejected candidate still had to be
written somewhere to be rejected.

**Honest expectation on CPU.** §3.5 measured single-sequence speculation at
**0.58x — slower** — on this CPU with a 360M/135M pair, because the models are
too close in cost and the draft's K passes are serial. Batched verification
removes the *per-sequence* multiplication of target passes, which is what made
speculation and batching mutually exclusive, but it does not change that
arithmetic. The phase's own definition of done asks for a GPU run to confirm or
refute the 2-3x claim, which the installed CPU-only torch build cannot produce
-- a package limitation rather than a hardware one, since the machine does have
a CUDA GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import torch
from transformers.cache_utils import DynamicCache

from memforge_ml.kernels.batched_decode import BatchedDecodeBuffers, assemble_batch
from memforge_ml.speculative.rejection_sampler import RejectionSampler


@dataclass
class DraftState:
    """One sequence's draft-model cache and how far it has been fed."""

    cache: DynamicCache = field(default_factory=DynamicCache)
    #: Tokens of the sequence already written into `cache`.
    position: int = 0

    def reset(self) -> None:
        self.cache = DynamicCache()
        self.position = 0

    def rollback_to(self, position: int) -> None:
        """
        Forget everything the draft conditioned on past `position`.

        After a round, the draft has run ahead by all K candidates while only
        the accepted prefix is real. Without this the draft would keep
        generating from a continuation the target rejected -- its proposals
        would get worse every round and the acceptance rate would collapse,
        quietly, with correct output throughout, since rejection sampling
        keeps the *output* right however bad the proposals are.

        Rolling back only as far as the accepted prefix, rather than all the
        way, keeps the accepted candidates' draft KV. That is a narrower
        catch-up pass next round, not a cheaper one: the catch-up is a single
        pass whatever its width.
        """
        if position < 0:
            raise ValueError(f"cannot roll a draft back to position {position}")
        extra = self.position - position
        if extra > 0:
            # Negative argument: the positive form is deprecated in this
            # transformers and removed in 5.18.
            self.cache.crop(-extra)
            self.position = position
        elif extra < 0:
            raise ValueError(
                f"draft is at {self.position}, behind the requested {position}"
            )


@dataclass
class SpeculativeRound:
    """What one speculative step produced for one sequence."""

    seq_id: int
    candidates: List[int]
    accepted: List[int]
    is_draft: List[bool]
    num_accepted_draft: int

    @property
    def num_confirmed(self) -> int:
        return len(self.accepted)


def _probs_from_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Distribution over the vocabulary for one position.

    At temperature 0 this is a point mass on the argmax rather than a division
    by zero, which is what makes greedy speculation reduce exactly to greedy
    decoding -- the property the equivalence test relies on.
    """
    if temperature <= 0.0:
        out = torch.zeros_like(logits)
        out[int(logits.argmax())] = 1.0
        return out
    return torch.softmax(logits / temperature, dim=-1)


class GraphedDraftState:
    """
    A `DraftState` whose single-token decode step is a replayed CUDA graph.

    Why: at batch 1 on a GPU, an eager Hugging Face decode step is bound by
    Python and kernel-launch overhead, not by arithmetic. Measured on an A100,
    Qwen3-0.6B costs 28.8 ms/token eager but only 7.6 ms of actual GPU time --
    63 % of the 8B target's step, so a "cheap" draft was not cheap and
    speculation lost (0.4-0.7x). One replayed graph costs 6.5 ms.

    The cache is a `StaticCache`, so tensor addresses never move. Rollback is
    just moving `position` back: stale entries past it are masked by the causal
    mask derived from `cache_position`, then overwritten.
    """

    def __init__(self, pool: "GraphedDraftPool", slot: int):
        self.pool = pool
        self.slot = slot
        self.position = 0
        self.cache = pool._caches[slot]

    def reset(self) -> None:
        self.position = 0

    def rollback_to(self, position: int) -> None:
        if position < 0:
            raise ValueError(f"cannot roll a draft back to position {position}")
        if position > self.position:
            raise ValueError(
                f"draft is at {self.position}, behind the requested {position}"
            )
        self.position = position


class GraphedDraftPool:
    """
    A fixed set of draft slots, each a static cache plus one captured graph.

    Slots (not per-sequence graphs) because capture is expensive and a
    sequence's lifetime is short next to the engine's. Graphs share one memory
    pool; that is safe because replays are serialised and the logits are
    cloned out of the shared output buffer before the next replay.
    """

    def __init__(self, model, max_len: int, num_slots: int, device: str):
        from transformers.cache_utils import StaticCache

        if not str(device).startswith("cuda"):
            raise ValueError("a CUDA graph draft needs a CUDA device")
        impl = getattr(model.config, "_attn_implementation", None)
        if impl != "sdpa":
            # The graph takes a boolean mask; only sdpa interprets that
            # correctly (the eager backend wants an additive float mask and
            # silently attends to the wrong keys).
            raise ValueError(f"CUDA-graph draft needs attn_implementation='sdpa', got {impl!r}")
        self.model = model
        self.device = device
        self.max_len = max_len
        self._dtype = next(model.parameters()).dtype
        self._mem_pool = torch.cuda.graph_pool_handle()
        self._caches = [
            StaticCache(
                config=model.config, max_batch_size=1, max_cache_len=max_len,
                device=device, dtype=self._dtype,
            )
            for _ in range(num_slots)
        ]
        self._tok = [torch.zeros((1, 1), dtype=torch.long, device=device) for _ in range(num_slots)]
        self._cp = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(num_slots)]
        # The causal mask must be a *tensor input* of the graph. Left to the
        # model, it is derived from the capture-time position on the host and
        # frozen into the graph, so every replay attends to the wrong keys
        # (measured: logits off by 16, different argmax).
        self._mask = [torch.zeros((1, 1, 1, max_len), dtype=torch.bool, device=device) for _ in range(num_slots)]
        self._arange = torch.arange(max_len, device=device)
        self._len_src = [torch.zeros((), dtype=torch.long, device=device) for _ in range(num_slots)]
        self._graph: List = [None] * num_slots
        self._out: List = [None] * num_slots
        self._free = list(range(num_slots))

    def acquire(self) -> GraphedDraftState:
        if not self._free:
            raise RuntimeError("no free draft slot; raise num_slots to max_num_seqs")
        slot = self._free.pop()
        return GraphedDraftState(self, slot)

    def release(self, state: GraphedDraftState) -> None:
        state.reset()
        self._free.append(state.slot)

    def _set_length(self, slot: int, position: int) -> None:
        """
        Point the cache's own write cursor at `position`.

        transformers' `StaticLayer` ignores the `cache_position` argument and
        writes at an internal `cumulative_length` that it bumps on every
        update. Warmup, capture, replay and rollback all move it, so without
        this the draft silently writes and reads the wrong slots. One batched
        copy keeps it to a single launch rather than one per layer.
        """
        layers = self._caches[slot].layers
        if all(getattr(l, "is_initialized", False) for l in layers):
            src = self._len_src[slot]
            src.fill_(position)
            torch._foreach_copy_([l.cumulative_length for l in layers], [src] * len(layers))
        else:
            for l in layers:
                l.cumulative_length.fill_(position)

    @torch.no_grad()
    def _step_fn(self, slot: int):
        return self.model(
            self._tok[slot], past_key_values=self._caches[slot],
            cache_position=self._cp[slot], attention_mask=self._mask[slot], use_cache=True,
        ).logits

    @torch.no_grad()
    def _capture(self, slot: int) -> None:
        # Warmup executes real steps. They must not touch live KV: park them on
        # the last position, which a real step overwrites before it can read it
        # (a step writes K/V at `cache_position` and only attends to <= it).
        # Capturing at position 0 corrupted the prompt's first KV entry.
        self._cp[slot].fill_(self.max_len - 1)
        self._mask[slot].copy_(self._arange <= (self.max_len - 1))
        self._tok[slot].zero_()
        self._set_length(slot, self.max_len - 1)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._set_length(slot, self.max_len - 1)
                self._step_fn(slot)
        torch.cuda.current_stream().wait_stream(stream)
        self._set_length(slot, self.max_len - 1)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._mem_pool):
            self._out[slot] = self._step_fn(slot)
        self._graph[slot] = graph

    @torch.no_grad()
    def step(self, state: GraphedDraftState, token: int) -> torch.Tensor:
        """Feed one token at `state.position`; return its logits."""
        slot = state.slot
        if state.position >= self.max_len:
            raise ValueError(f"draft context {state.position} exceeds {self.max_len}")
        if self._graph[slot] is None:
            self._capture(slot)
        self._tok[slot].fill_(token)
        self._cp[slot].fill_(state.position)
        self._mask[slot].copy_(self._arange <= state.position)
        self._set_length(slot, state.position)
        self._graph[slot].replay()
        state.position += 1
        return self._out[slot][0, -1].clone()

    @torch.no_grad()
    def prefill(self, state: GraphedDraftState, tokens: Sequence[int]) -> torch.Tensor:
        """Feed several tokens eagerly (variable width); return the last logits."""
        n = len(tokens)
        if state.position + n > self.max_len:
            raise ValueError(f"draft context {state.position + n} exceeds {self.max_len}")
        ids = torch.tensor([list(tokens)], dtype=torch.long, device=self.device)
        cp = torch.arange(state.position, state.position + n, device=self.device)
        self._set_length(state.slot, state.position)
        out = self.model(
            ids, past_key_values=self._caches[state.slot], cache_position=cp, use_cache=True
        )
        state.position += n
        return out.logits[0, -1]


@torch.no_grad()
def propose_draft_tokens(
    draft_model,
    state: DraftState,
    tokens: Sequence[int],
    lookahead: int,
    temperature: float,
    device: str,
) -> Tuple[List[int], List[torch.Tensor]]:
    """
    Run the draft model forward `lookahead` times for one sequence.

    `tokens` is the full confirmed token list. Everything the draft has not seen
    yet is fed in one pass before the proposal loop starts, which matters after
    a round where several tokens were confirmed at once: feeding them one at a
    time would cost a forward pass each.

    Returns the candidate tokens and the draft's own distribution at each step,
    which rejection sampling needs as `q(x)`.
    """
    candidates: List[int] = []
    probs: List[torch.Tensor] = []

    if isinstance(state, GraphedDraftState):
        return _propose_graphed(state, tokens, lookahead, temperature)

    pending = list(tokens[state.position :])
    if not pending:
        # Nothing new to condition on. Caller must not ask for candidates in
        # this state; it means the draft is already ahead of the sequence.
        raise ValueError(
            f"draft is at position {state.position} with only {len(tokens)} tokens"
        )

    ids = torch.tensor([pending], dtype=torch.long, device=device)
    out = draft_model(input_ids=ids, past_key_values=state.cache, use_cache=True)
    state.position = len(tokens)

    for _ in range(lookahead):
        step_probs = _probs_from_logits(out.logits[0, -1], temperature)
        token = int(step_probs.argmax()) if temperature <= 0.0 else int(
            torch.multinomial(step_probs, num_samples=1).item()
        )
        candidates.append(token)
        probs.append(step_probs)

        out = draft_model(
            input_ids=torch.tensor([[token]], dtype=torch.long, device=device),
            past_key_values=state.cache,
            use_cache=True,
        )
        state.position += 1

    return candidates, probs


@torch.no_grad()
def verify_candidates_batched(
    target_model,
    pool,
    buffers: BatchedDecodeBuffers,
    block_tables: Sequence[torch.Tensor],
    context_lengths: Sequence[int],
    query_tokens: Sequence[Sequence[int]],
    device: str,
) -> Tuple[List[List[torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]], int]:
    """
    One target forward pass over every sequence's whole candidate run.

    `query_tokens[i]` is `[last_confirmed, d1, ..., dK]` for sequence *i*, all of
    the same length, which is what makes this a plain padded batch rather than
    the varlen problem that blocks prefill/decode fusion.

    Returns per-sequence logits (one tensor per query position), the new K/V per
    layer so the caller can scatter the accepted prefix back into the pool, and
    the padded context width.
    """
    batch = len(context_lengths)
    if batch == 0:
        raise ValueError("verify_candidates_batched requires at least one sequence")
    widths = {len(q) for q in query_tokens}
    if len(widths) != 1:
        raise ValueError(f"all sequences must propose the same count, got {widths}")
    query_len = widths.pop()
    if query_len < 1:
        raise ValueError("each sequence needs at least one query token")

    layer_data, context_mask, padded = assemble_batch(
        pool, buffers, block_tables, context_lengths
    )
    cache = DynamicCache(ddp_cache_data=layer_data)

    input_ids = torch.tensor(list(query_tokens), dtype=torch.long, device=device)

    # Context columns keep their left-padded mask; the query columns are all
    # real tokens, so they are unmasked here. Causality *among* the candidates
    # is not expressed by this mask -- the model's own causal mask construction
    # supplies it, and `test_candidate_logits_ignore_later_candidates` verifies
    # it by perturbation rather than by trusting that it happens.
    attention_mask = torch.zeros(
        (batch, padded + query_len), dtype=torch.long, device=device
    )
    attention_mask[:, :padded] = context_mask
    attention_mask[:, padded:] = 1

    # Absolute positions per row, so RoPE sees each sequence's true offsets
    # rather than the padded batch's.
    position_ids = torch.stack(
        [
            torch.arange(ctx, ctx + query_len, dtype=torch.long, device=device)
            for ctx in context_lengths
        ]
    )
    # Passed for API correctness, and it is the value transformers would infer
    # anyway. It is deliberately not relied on: mutating it (reversing it,
    # collapsing it, shifting it) changes nothing observable, so the causal
    # structure of the query block rests on the mask and `position_ids`, not on
    # this argument.
    cache_position = torch.arange(padded, padded + query_len, device=device)

    out = target_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        cache_position=cache_position,
        use_cache=True,
    )

    per_sequence_logits = [
        [out.logits[row, pos] for pos in range(query_len)] for row in range(batch)
    ]
    new_kv = [
        (
            cache.layers[layer].keys[:, :, -query_len:, :],
            cache.layers[layer].values[:, :, -query_len:, :],
        )
        for layer in range(pool.spec.num_layers)
    ]
    return per_sequence_logits, new_kv, padded


def resolve_round(
    seq_id: int,
    candidates: List[int],
    draft_probs: List[torch.Tensor],
    target_logits: List[torch.Tensor],
    temperature: float,
) -> SpeculativeRound:
    """
    Decide how much of one sequence's candidate run survives.

    Delegates entirely to the existing `RejectionSampler` rather than
    reimplementing the acceptance rule, so batched and single-sequence
    speculation cannot drift apart in their statistics -- the losslessness
    argument covers both because there is only one implementation.

    `target_logits[i]` is the distribution for the token at candidate slot *i*,
    and the K+1'th entry is the bonus token, which `verify_tokens` picks up on
    its own when every candidate was accepted.

    Greedy decoding falls out of this unchanged: at temperature 0 both
    distributions are point masses, so an agreeing candidate has
    `p/q = 1` and is always accepted, and a disagreeing one has `p = 0`, is
    always rejected, and resampling from `clamp(p - q)` returns the target's
    own argmax. That is why speculation at temperature 0 is byte-identical to
    plain greedy decoding rather than merely close to it.
    """
    # Same tokenizer, different padded vocabulary (e.g. Qwen2.5-7B: 152064 rows
    # vs 151936 for its smaller siblings). The extra rows are padding no
    # tokenizer can emit, so the target's distribution is restricted to the
    # width the draft covers *before* the softmax; comparing distributions of
    # different widths is what used to crash here.
    if draft_probs:
        width = draft_probs[0].shape[-1]
        target_logits = [logit[..., :width] for logit in target_logits]
    target_probs = [_probs_from_logits(logit, temperature) for logit in target_logits]

    accepted, is_draft, num_accepted = RejectionSampler.verify_tokens(
        candidates, draft_probs, target_probs
    )

    return SpeculativeRound(
        seq_id=seq_id,
        candidates=list(candidates),
        accepted=list(accepted),
        is_draft=list(is_draft),
        num_accepted_draft=num_accepted,
    )


def scatter_accepted_kv(
    pool,
    block_table: torch.Tensor,
    context_length: int,
    new_kv: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    row: int,
    num_confirmed: int,
) -> None:
    """
    Write only the confirmed prefix of a candidate run into the pool.

    The target pass computed K+1 positions' worth of K/V; the rejected tail is
    simply not written, which is why rejection costs nothing to undo. The blocks
    it would have occupied were reserved by `lookahead_slots` and are reused by
    the next round.
    """
    if num_confirmed <= 0:
        return
    # Addressing depends only on the token range, not the layer, so it is
    # computed once rather than once per layer -- the same reason the decode
    # path hoists it.
    physical, slots = pool.slot_indices(
        block_table, context_length, context_length + num_confirmed
    )
    for layer_idx, (keys, values) in enumerate(new_kv):
        pool.write(
            layer_idx,
            physical,
            slots,
            keys[row : row + 1, :, :num_confirmed, :],
            values[row : row + 1, :, :num_confirmed, :],
        )


def _propose_graphed(state, tokens, lookahead, temperature):
    """`propose_draft_tokens` for a graph-backed draft: same maths, replayed steps."""
    pool = state.pool
    pending = list(tokens[state.position :])
    if not pending:
        raise ValueError(
            f"draft is at position {state.position} with only {len(tokens)} tokens"
        )
    logits = (
        pool.step(state, pending[0]) if len(pending) == 1 else pool.prefill(state, pending)
    )
    candidates: List[int] = []
    probs: List[torch.Tensor] = []
    for _ in range(lookahead):
        step_probs = _probs_from_logits(logits, temperature)
        token = int(step_probs.argmax()) if temperature <= 0.0 else int(
            torch.multinomial(step_probs, num_samples=1).item()
        )
        candidates.append(token)
        probs.append(step_probs)
        logits = pool.step(state, token)
    return candidates, probs
