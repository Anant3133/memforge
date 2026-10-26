"""
Tests for batched speculative verification (Phase D bullet 2).

Two properties carry the phase:

* **Batching is invisible.** Verifying N sequences in one padded target pass
  must decide exactly what verifying each one alone would decide. Left padding,
  per-row `position_ids` and the paged gather all have to line up, and every one
  of them is a silent-corruption bug if it does not -- a wrong context read
  produces fluent, confident, wrong text rather than a crash. The reference here
  is a per-sequence `DynamicCache` pass, which shares no code with the batched
  path.
* **Rejection is lossless.** At temperature 0 a speculative round must confirm
  exactly the target's own greedy continuation, no matter what the draft
  proposed, including when the draft is adversarially wrong.

Logit comparisons use `allclose`, not `torch.equal`: batched and single-row
matmuls reduce in a different order on CPU, so bit-equality is not available
here the way it is for the pure-memory kernels in `test_kernels.py`. The
*decisions* -- argmax per position, and the confirmed token list -- are compared
exactly, since those are what the output depends on.
"""

from __future__ import annotations

import pytest
import torch
from transformers.cache_utils import DynamicCache

from memforge_ml.kernels.batched_decode import BatchedDecodeBuffers
from memforge_ml.runtime.batched_speculative import (
    DraftState,
    SpeculativeRound,
    propose_draft_tokens,
    resolve_round,
    scatter_accepted_kv,
    verify_candidates_batched,
)
from memforge_ml.runtime.kv_pool import ModelKVSpec, PagedKVPool

BLOCK = 4
VOCAB = 256


# --------------------------------------------------------------- fixtures

def make_pool(model, num_blocks: int = 128):
    config = model.config
    spec = ModelKVSpec(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        dtype=torch.float32,
        device="cpu",
    )
    return PagedKVPool(spec, num_blocks=num_blocks, block_size=BLOCK)


@torch.no_grad()
def prefill_into_pool(model, pool, tokens, block_table):
    """
    Run a real prefill for one sequence and scatter its K/V into the pool.

    Deliberately a plain `DynamicCache` pass: the pool must end up holding the
    KV that an ordinary forward pass would have produced, so the batched
    verification under test is reading genuine cache contents and not something
    this helper and the implementation agreed on together.
    """
    ids = torch.tensor([tokens], dtype=torch.long)
    cache = DynamicCache()
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    physical, slots = pool.slot_indices(block_table, 0, len(tokens))
    for layer in range(pool.spec.num_layers):
        pool.write(
            layer,
            physical,
            slots,
            cache.layers[layer].keys,
            cache.layers[layer].values,
        )
    return out.logits


@torch.no_grad()
def reference_verify(model, context_tokens, query_tokens):
    """Per-sequence reference: one pass over context+queries, no padding, no pool."""
    ids = torch.tensor([list(context_tokens) + list(query_tokens)], dtype=torch.long)
    logits = model(input_ids=ids, use_cache=False).logits
    return [logits[0, -len(query_tokens) + i] for i in range(len(query_tokens))]


def build_batch(model, pool, cached, lookahead: int = 0):
    """
    Prefill each sequence's *cached* tokens into its own blocks.

    `cached[row]` is exactly what the pool holds, so `context_length` is
    `len(cached[row])` and the sequence's last confirmed token is the one at
    index `len(cached[row])` -- deliberately **not** in the pool. That is the
    engine's own convention (`cache.position` counts written tokens while
    `tokens[-1]` is still waiting to be fed), and getting it wrong by one
    position makes a token attend to itself twice, which reads as plausible
    output rather than as an error.

    `lookahead` reserves blocks for the K+1 candidates on top of the context,
    mirroring `BlockAllocator::ensure_capacity_with_lookahead`: a candidate has
    to have somewhere to be written before it can be rejected.
    """
    tables, lengths = [], []
    next_block = 0
    for tokens in cached:
        span = len(tokens) + lookahead
        n_blocks = (span + BLOCK - 1) // BLOCK
        table = PagedKVPool.block_table_tensor(
            list(range(next_block, next_block + n_blocks)), "cpu"
        )
        next_block += n_blocks
        prefill_into_pool(model, pool, tokens, table)
        tables.append(table)
        lengths.append(len(tokens))
    return tables, lengths


# ------------------------------------------------- batched == per-sequence

@pytest.mark.parametrize(
    "contexts",
    [
        # Equal lengths: no padding involved, isolates everything else.
        [[5, 9, 13, 21], [7, 3, 19, 2]],
        # Ragged, crossing block boundaries in different places. This is the
        # case that catches a position_ids or left-padding mistake.
        [[5, 9, 13, 21, 44, 8, 2], [7, 3], [11, 15, 19, 23, 27]],
        # A single sequence must still work through the batched path.
        [[31, 17, 5]],
    ],
)
def test_batched_verification_matches_per_sequence(tiny_model, contexts):
    lookahead = 3
    pool = make_pool(tiny_model)
    tables, lengths = build_batch(tiny_model, pool, contexts, lookahead + 1)

    # query[0] is the last confirmed token, which is not in the pool; the rest
    # are the draft's candidates. Candidates differ per row on purpose: a bug
    # that collapsed rows together would still pass if every row proposed the
    # same tokens.
    queries = [
        [(60 + row) % VOCAB] + [(40 + 7 * row + i) % VOCAB for i in range(lookahead)]
        for row in range(len(contexts))
    ]

    buffers = BatchedDecodeBuffers(pool.spec.num_layers, block_size=BLOCK)
    logits, new_kv, padded = verify_candidates_batched(
        tiny_model, pool, buffers, tables, lengths, queries, "cpu"
    )

    assert len(logits) == len(contexts)
    assert padded >= max(lengths)
    assert len(new_kv) == pool.spec.num_layers

    for row, (ctx, query) in enumerate(zip(contexts, queries)):
        expected = reference_verify(tiny_model, ctx, query)
        assert len(logits[row]) == len(query)
        for pos, (got, want) in enumerate(zip(logits[row], expected)):
            assert torch.allclose(got, want, atol=2e-4), (
                f"row {row} position {pos} diverged: "
                f"max |delta| = {(got - want).abs().max():.2e}"
            )
            assert int(got.argmax()) == int(want.argmax()), (
                f"row {row} position {pos} would pick a different token"
            )


def test_batched_verification_new_kv_has_query_width(tiny_model):
    """The returned K/V covers exactly the query positions, ready to scatter."""
    pool = make_pool(tiny_model)
    contexts = [[5, 9, 13, 21, 44], [7, 3, 19]]
    tables, lengths = build_batch(tiny_model, pool, contexts, 3)
    queries = [[60, 50, 51] for _ in contexts]

    buffers = BatchedDecodeBuffers(pool.spec.num_layers, block_size=BLOCK)
    _, new_kv, _ = verify_candidates_batched(
        tiny_model, pool, buffers, tables, lengths, queries, "cpu"
    )
    for keys, values in new_kv:
        assert keys.shape[0] == len(contexts)
        assert keys.shape[2] == 3
        assert keys.shape == values.shape


def test_batched_verification_rejects_ragged_candidate_counts(tiny_model):
    """
    Uniform width is the precondition that makes this a plain padded batch.

    A caller that let one sequence propose fewer candidates would silently
    misalign every row after it, so this is refused rather than padded over.
    """
    pool = make_pool(tiny_model)
    tables, lengths = build_batch(tiny_model, pool, [[5, 9, 13], [7, 3, 19]], 3)
    buffers = BatchedDecodeBuffers(pool.spec.num_layers, block_size=BLOCK)

    with pytest.raises(ValueError, match="same count"):
        verify_candidates_batched(
            tiny_model, pool, buffers, tables, lengths,
            [[60, 40, 41], [61, 42]], "cpu",
        )


def test_batched_verification_rejects_empty_batch(tiny_model):
    pool = make_pool(tiny_model)
    buffers = BatchedDecodeBuffers(pool.spec.num_layers, block_size=BLOCK)
    with pytest.raises(ValueError, match="at least one sequence"):
        verify_candidates_batched(tiny_model, pool, buffers, [], [], [], "cpu")


def test_candidate_logits_ignore_later_candidates(tiny_model):
    """
    The candidate run must be causal: position *i* judges `d(i+1)` and so must
    not see `d(i+2)` onwards.

    This is checked by perturbation, not against a reference, because it is the
    property that cannot be allowed to hold only by coincidence. If verification
    ran bidirectionally over the K+1 query positions, every accept/reject
    decision would be made against a distribution the target model would never
    produce -- and the output would still be fluent, which is exactly what makes
    it dangerous. An earlier bug in this codebase had precisely this shape:
    `create_causal_mask` returned `None` under `allow_is_causal_skip` and
    prefill silently ran bidirectional.
    """
    pool = make_pool(tiny_model)
    context = [5, 9, 13, 21, 44]
    tables, lengths = build_batch(tiny_model, pool, [context], 4)
    buffers = BatchedDecodeBuffers(pool.spec.num_layers, block_size=BLOCK)

    def logits_for(candidates):
        got, _, _ = verify_candidates_batched(
            tiny_model, pool, buffers, tables, lengths,
            [[60] + candidates], "cpu",
        )
        return got[0]

    base = logits_for([70, 71, 72])
    later_changed = logits_for([70, 200, 201])   # same d1, different d2/d3
    first_changed = logits_for([200, 71, 72])    # different d1, same d2/d3

    # Changing d2/d3 cannot move the position that judges d1, nor the one that
    # judges d2 (whose input is d1).
    assert torch.equal(base[0], later_changed[0]), "position 0 saw a later candidate"
    assert torch.equal(base[1], later_changed[1]), "position 1 saw a later candidate"
    # Changing d1 cannot move the position that judges d1 either, since d1 is
    # that position's *output*, not its input.
    assert torch.equal(base[0], first_changed[0]), "position 0 saw its own candidate"

    # Non-vacuity: an earlier candidate must move the later positions, or the
    # equalities above would be measuring constants rather than causality.
    assert not torch.equal(base[1], first_changed[1]), "d1 did not reach position 1"
    assert not torch.equal(base[2], later_changed[2]), "d2 did not reach position 2"


def test_padding_does_not_change_a_sequence_result(tiny_model):
    """
    A sequence's verification must not depend on who it was batched with.

    Batched alongside a much longer sequence, a short context picks up a wide
    band of left padding. If that padding leaked into attention, results would
    shift with batch composition -- a bug that only appears under real mixed
    traffic and never in a single-sequence test.
    """
    short = [7, 3, 19]
    # Long enough to land in a wider padding bucket than `short` does;
    # `BatchedDecodeBuffers` rounds the padded width up in block groups,
    # so a merely-longer sequence can share a bucket and test nothing.
    long_ = [11, 15, 19, 23, 27, 31, 2, 44, 8, 5, 9, 13, 21, 7, 3, 17,
             29, 33, 6, 10]
    lookahead = 3
    queries_short = [60, 40, 41, 42]

    alone = make_pool(tiny_model)
    tables_a, lengths_a = build_batch(tiny_model, alone, [short], lookahead + 1)
    buffers_a = BatchedDecodeBuffers(alone.spec.num_layers, block_size=BLOCK)
    solo, _, solo_pad = verify_candidates_batched(
        tiny_model, alone, buffers_a, tables_a, lengths_a, [queries_short], "cpu"
    )

    together = make_pool(tiny_model)
    tables_b, lengths_b = build_batch(
        tiny_model, together, [short, long_], lookahead + 1
    )
    buffers_b = BatchedDecodeBuffers(together.spec.num_layers, block_size=BLOCK)
    mixed, _, mixed_pad = verify_candidates_batched(
        tiny_model, together, buffers_b, tables_b, lengths_b,
        [queries_short, [61, 50, 51, 52]], "cpu",
    )

    # The point of the test: the second batch really is padded wider.
    assert mixed_pad > solo_pad

    for pos, (a, b) in enumerate(zip(solo[0], mixed[0])):
        assert torch.allclose(a, b, atol=2e-4), (
            f"position {pos} moved when batched with a longer sequence: "
            f"max |delta| = {(a - b).abs().max():.2e}"
        )
        assert int(a.argmax()) == int(b.argmax())


# ------------------------------------------------------------- acceptance

def one_hot(token: int, vocab: int = VOCAB) -> torch.Tensor:
    probs = torch.zeros(vocab)
    probs[token] = 1.0
    return probs


def logits_favouring(token: int, vocab: int = VOCAB) -> torch.Tensor:
    """Logits whose argmax is `token`, for driving temperature-0 decisions."""
    logits = torch.zeros(vocab)
    logits[token] = 10.0
    return logits


def test_greedy_round_accepts_every_correct_candidate():
    """A draft that guesses right gets all K accepted plus the free bonus."""
    candidates = [11, 22, 33]
    round_ = resolve_round(
        seq_id=1,
        candidates=candidates,
        draft_probs=[one_hot(t) for t in candidates],
        target_logits=[logits_favouring(t) for t in candidates + [44]],
        temperature=0.0,
    )
    assert round_.accepted == [11, 22, 33, 44]
    assert round_.num_accepted_draft == 3
    # K+1 tokens out of one target pass -- the whole point of the phase.
    assert round_.num_confirmed == len(candidates) + 1
    assert round_.is_draft == [True, True, True, False]


def test_greedy_round_truncates_at_first_wrong_candidate():
    """
    A wrong candidate ends the run and is replaced by the target's own token.

    Everything after it is discarded even if it would have been right, because
    those candidates were conditioned on a token the target did not pick.
    """
    round_ = resolve_round(
        seq_id=1,
        candidates=[11, 22, 33],
        draft_probs=[one_hot(11), one_hot(22), one_hot(33)],
        target_logits=[
            logits_favouring(11),
            logits_favouring(99),  # target disagrees here
            logits_favouring(33),
            logits_favouring(44),
        ],
        temperature=0.0,
    )
    assert round_.accepted == [11, 99]
    assert round_.num_accepted_draft == 1
    assert round_.is_draft == [True, False]


def test_greedy_round_with_useless_draft_still_yields_target_token():
    """
    Worst case: every candidate wrong. The round still confirms one token, and
    it is the target's. Speculation can waste work but never corrupt output.
    """
    round_ = resolve_round(
        seq_id=1,
        candidates=[11, 22, 33],
        draft_probs=[one_hot(11), one_hot(22), one_hot(33)],
        target_logits=[logits_favouring(77)] * 4,
        temperature=0.0,
    )
    assert round_.accepted == [77]
    assert round_.num_accepted_draft == 0
    assert round_.is_draft == [False]


def test_greedy_round_confirms_target_argmax_sequence(tiny_model):
    """
    End to end on a real model: a greedy round confirms a prefix of the target's
    own greedy continuation, whatever the draft proposed.

    This is the losslessness property stated against real logits rather than
    hand-built one-hots -- the draft here is deliberately garbage.
    """
    pool = make_pool(tiny_model)
    context = [5, 9, 13, 21, 44]
    tables, lengths = build_batch(tiny_model, pool, [context], 4)

    garbage = [200, 201, 202]
    queries = [[60] + garbage]

    buffers = BatchedDecodeBuffers(pool.spec.num_layers, block_size=BLOCK)
    logits, _, _ = verify_candidates_batched(
        tiny_model, pool, buffers, tables, lengths, queries, "cpu"
    )

    round_ = resolve_round(
        seq_id=0,
        candidates=garbage,
        draft_probs=[one_hot(t) for t in garbage],
        target_logits=logits[0],
        temperature=0.0,
    )

    # A garbage draft is rejected at the first candidate, so exactly one token
    # is confirmed and it must be the target's greedy choice.
    assert round_.num_accepted_draft == 0
    assert round_.accepted == [int(logits[0][0].argmax())]


def test_sampled_round_never_exceeds_lookahead_plus_one():
    """Invariant the scheduler's K+1 reservation depends on."""
    torch.manual_seed(0)
    lookahead = 4
    for _ in range(25):
        probs = [torch.rand(VOCAB).softmax(dim=-1) for _ in range(lookahead)]
        candidates = [int(torch.multinomial(p, 1)) for p in probs]
        round_ = resolve_round(
            seq_id=1,
            candidates=candidates,
            draft_probs=probs,
            target_logits=[torch.randn(VOCAB) for _ in range(lookahead + 1)],
            temperature=1.0,
        )
        assert 1 <= round_.num_confirmed <= lookahead + 1
        assert round_.num_accepted_draft <= lookahead
        assert len(round_.is_draft) == round_.num_confirmed


# ----------------------------------------------------------- KV write-back

def test_scatter_writes_only_the_confirmed_prefix(tiny_model):
    """
    Rejected candidates must leave no trace in the pool.

    The target pass computed K+1 positions, but only the accepted prefix may be
    written: the rest of the reservation has to look untouched so the next
    round overwrites it. A scatter that wrote all K+1 would corrupt the context
    of every subsequent token.
    """
    pool = make_pool(tiny_model)
    context = [5, 9, 13, 21, 44]
    tables, lengths = build_batch(tiny_model, pool, [context], 4)
    table, ctx_len = tables[0], lengths[0]

    buffers = BatchedDecodeBuffers(pool.spec.num_layers, block_size=BLOCK)
    _, new_kv, _ = verify_candidates_batched(
        tiny_model, pool, buffers, [table], [ctx_len],
        [[60, 50, 51, 52]], "cpu",
    )

    # Mark the whole reservation so an unwanted write is visible.
    physical, slots = pool.slot_indices(table, ctx_len, ctx_len + 4)
    pool.key_pool[:, physical, slots] = 0.0
    pool.value_pool[:, physical, slots] = 0.0

    scatter_accepted_kv(pool, table, ctx_len, new_kv, row=0, num_confirmed=2)

    written_p, written_s = pool.slot_indices(table, ctx_len, ctx_len + 2)
    untouched_p, untouched_s = pool.slot_indices(table, ctx_len + 2, ctx_len + 4)
    assert pool.key_pool[:, written_p, written_s].abs().sum() > 0, "confirmed KV missing"
    assert torch.equal(
        pool.key_pool[:, untouched_p, untouched_s],
        torch.zeros_like(pool.key_pool[:, untouched_p, untouched_s]),
    ), "rejected candidate's KV leaked into the pool"


def test_scatter_round_trips_through_the_pool(tiny_model):
    """What is scattered in must gather back out bit-identically."""
    pool = make_pool(tiny_model)
    context = [5, 9, 13, 21, 44, 8]
    tables, lengths = build_batch(tiny_model, pool, [context], 3)
    table, ctx_len = tables[0], lengths[0]

    buffers = BatchedDecodeBuffers(pool.spec.num_layers, block_size=BLOCK)
    _, new_kv, _ = verify_candidates_batched(
        tiny_model, pool, buffers, [table], [ctx_len],
        [[60, 50, 51]], "cpu",
    )
    scatter_accepted_kv(pool, table, ctx_len, new_kv, row=0, num_confirmed=3)

    physical, slots = pool.slot_indices(table, ctx_len, ctx_len + 3)
    for layer, (keys, values) in enumerate(new_kv):
        got_k, got_v = pool.gather(layer, physical, slots)
        assert torch.equal(got_k, keys[0:1].contiguous())
        assert torch.equal(got_v, values[0:1].contiguous())


def test_scatter_of_nothing_is_a_no_op(tiny_model):
    """num_confirmed=0 cannot happen today, but must not write if it ever does."""
    pool = make_pool(tiny_model)
    tables, lengths = build_batch(tiny_model, pool, [[5, 9, 13]], 2)
    before = pool.key_pool.clone()
    scatter_accepted_kv(pool, tables[0], lengths[0], [], row=0, num_confirmed=0)
    assert torch.equal(pool.key_pool, before)


# ------------------------------------------------------------- proposal

def test_proposal_returns_lookahead_candidates_and_probs(tiny_draft_model):
    state = DraftState()
    tokens = [5, 9, 13, 21]
    candidates, probs = propose_draft_tokens(
        tiny_draft_model, state, tokens, lookahead=4, temperature=0.0, device="cpu"
    )
    assert len(candidates) == 4
    assert len(probs) == 4
    assert all(p.shape == (VOCAB,) for p in probs)
    # Fed the whole prompt plus each candidate it then conditioned on.
    assert state.position == len(tokens) + 4


def test_proposal_catches_up_multiple_confirmed_tokens_in_one_pass(tiny_draft_model):
    """
    After a round that confirmed several tokens, the draft is behind by all of
    them. Feeding them one per forward pass would cost the speculative win, so
    the catch-up is a single pass -- and must give the same result.
    """
    tokens = [5, 9, 13, 21]
    confirmed = [31, 7, 19]

    fresh = DraftState()
    torch.manual_seed(0)
    all_at_once, _ = propose_draft_tokens(
        tiny_draft_model, fresh, tokens + confirmed,
        lookahead=2, temperature=0.0, device="cpu",
    )

    staged = DraftState()
    torch.manual_seed(0)
    propose_draft_tokens(
        tiny_draft_model, staged, tokens, lookahead=0, temperature=0.0, device="cpu"
    )
    # The staged draft has seen the prompt; now it catches up on `confirmed`.
    staged_result, _ = propose_draft_tokens(
        tiny_draft_model, staged, tokens + confirmed,
        lookahead=2, temperature=0.0, device="cpu",
    )
    assert staged_result == all_at_once
    assert staged.position == fresh.position


def test_proposal_refuses_when_draft_is_already_current(tiny_draft_model):
    """
    Nothing new to condition on means the caller lost track of the draft's
    position; proposing anyway would condition on a stale distribution.
    """
    state = DraftState()
    tokens = [5, 9, 13]
    propose_draft_tokens(
        tiny_draft_model, state, tokens, lookahead=0, temperature=0.0, device="cpu"
    )
    with pytest.raises(ValueError, match="draft is at position"):
        propose_draft_tokens(
            tiny_draft_model, state, tokens, lookahead=2, temperature=0.0, device="cpu"
        )


def test_draft_state_reset_forgets_the_cache(tiny_draft_model):
    state = DraftState()
    propose_draft_tokens(
        tiny_draft_model, state, [5, 9, 13], lookahead=1, temperature=0.0, device="cpu"
    )
    assert state.position > 0
    state.reset()
    assert state.position == 0
    assert len(state.cache.layers) == 0


# ----------------------------------------------------- full round, batched

def test_full_speculative_round_over_a_batch(tiny_model, tiny_draft_model):
    """
    Propose, verify in one batched pass, accept, write back -- for three
    sequences at once, with the pool as the single source of truth throughout.
    """
    pool = make_pool(tiny_model)
    lookahead = 3
    contexts = [[5, 9, 13, 21, 44], [7, 3], [11, 15, 19, 23, 27, 31, 2]]
    tables, lengths = build_batch(tiny_model, pool, contexts, lookahead + 1)

    # Each sequence's last confirmed token is the one past its cached context,
    # which is what the draft conditions on and what the target re-feeds.
    last_tokens = [(60 + row) % VOCAB for row in range(len(contexts))]

    drafts, proposals, draft_probs = [], [], []
    for tokens, last in zip(contexts, last_tokens):
        state = DraftState()
        candidates, probs = propose_draft_tokens(
            tiny_draft_model, state, tokens + [last], lookahead, 0.0, "cpu"
        )
        drafts.append(state)
        proposals.append(candidates)
        draft_probs.append(probs)

    assert all(len(c) == lookahead for c in proposals)

    queries = [[last] + cand for last, cand in zip(last_tokens, proposals)]
    buffers = BatchedDecodeBuffers(pool.spec.num_layers, block_size=BLOCK)
    logits, new_kv, _ = verify_candidates_batched(
        tiny_model, pool, buffers, tables, lengths, queries, "cpu"
    )

    rounds = [
        resolve_round(row, proposals[row], draft_probs[row], logits[row], 0.0)
        for row in range(len(contexts))
    ]

    for row, round_ in enumerate(rounds):
        assert isinstance(round_, SpeculativeRound)
        assert 1 <= round_.num_confirmed <= lookahead + 1
        # Sequences are decided independently: one row's rejection must not
        # truncate another's run.
        scatter_accepted_kv(
            pool, tables[row], lengths[row], new_kv, row, round_.num_confirmed
        )
        physical, slots = pool.slot_indices(
            tables[row], lengths[row], lengths[row] + round_.num_confirmed
        )
        assert pool.key_pool[:, physical, slots].abs().sum() > 0

    # One target forward pass covered every sequence. That is the phase's claim.
    assert len({len(q) for q in queries}) == 1
