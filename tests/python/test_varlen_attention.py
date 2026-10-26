"""
Equivalence tests for variable-length block-wise paged attention (Phase C).

This kernel is the thing that makes prefill/decode fusion possible, and it is
also the easiest place in the project to be confidently wrong: a mask off by
one position, or a partial final block read one slot too far, produces a
plausible number rather than an error. The resulting text is fluent and wrong.

So every test here compares against `reference_varlen_attention`, which
materialises each query token's visible context and runs one plain softmax over
it. It shares no code with the kernel -- no block loop, no online softmax --
so it cannot agree with the kernel by construction.

Tolerance is `atol=1e-5` on fp32 rather than `torch.equal`: the online softmax
sums in a different order than a single softmax does, so bit-equality is not
available and demanding it would be wrong. The mask *decisions* are exact and
are tested separately by construction (a masked key must contribute nothing at
all), because those are discrete and a tolerance would hide a real error.
"""

from __future__ import annotations

import math

import pytest
import torch

from memforge_ml.kernels.varlen_attention import (
    group_by_sequence,
    paged_attention_varlen,
    reference_varlen_attention,
)
from memforge_ml.runtime.kv_pool import ModelKVSpec, PagedKVPool

BLOCK = 4
ATOL = 1e-5


def make_pool(num_blocks=64, layers=2, kv_heads=2, head_dim=8, seed=0):
    spec = ModelKVSpec(
        num_layers=layers, num_kv_heads=kv_heads, head_dim=head_dim,
        dtype=torch.float32, device="cpu",
    )
    pool = PagedKVPool(spec, num_blocks=num_blocks, block_size=BLOCK)
    torch.manual_seed(seed)
    pool.key_pool.normal_()
    pool.value_pool.normal_()
    return pool


def tables_for(lengths, pool):
    """One block table per sequence, carved out of distinct physical blocks."""
    tables = []
    nxt = 0
    for length in lengths:
        n = (length + BLOCK - 1) // BLOCK
        tables.append(PagedKVPool.block_table_tensor(list(range(nxt, nxt + n)), "cpu"))
        nxt += n
    assert nxt <= pool.key_pool.shape[1], "not enough blocks for this layout"
    return tables


def make_query(total, num_heads=4, head_dim=8, seed=1):
    torch.manual_seed(seed)
    return torch.randn(1, num_heads, total, head_dim)


def assert_matches_reference(pool, query, tables, seq_ids, positions, layer=0):
    got = paged_attention_varlen(
        query, pool.key_pool, pool.value_pool, layer, tables,
        seq_ids, positions, BLOCK,
    )
    want = reference_varlen_attention(
        query, pool.key_pool, pool.value_pool, layer, tables,
        seq_ids, positions, BLOCK,
    )
    assert got.shape == want.shape == query.shape
    assert torch.isfinite(got).all(), "kernel produced nan or inf"
    delta = (got - want).abs().max()
    assert torch.allclose(got, want, atol=ATOL), (
        f"max |delta| = {delta:.3e} over seq_ids={list(seq_ids)} "
        f"positions={list(positions)}"
    )
    return got


# ------------------------------------------------------------- row grouping

def test_group_by_sequence_preserves_first_seen_order():
    assert group_by_sequence([2, 0, 2, 1, 0]) == [(2, [0, 2]), (0, [1, 4]), (1, [3])]


def test_group_by_sequence_handles_a_single_row():
    assert group_by_sequence([7]) == [(7, [0])]


# ---------------------------------------------------- decode-only equivalence

@pytest.mark.parametrize("lengths", [[1], [4], [5], [16], [7, 13], [3, 4, 5, 9]])
def test_decode_only_matches_reference(lengths):
    """
    One query token per sequence, each at the end of its context. This is the
    case the existing single-query kernel already covers, so agreement here
    says the packing and per-sequence block addressing are right before
    causality is involved at all.
    """
    pool = make_pool()
    tables = tables_for(lengths, pool)
    seq_ids = list(range(len(lengths)))
    positions = [n - 1 for n in lengths]
    query = make_query(len(lengths))
    assert_matches_reference(pool, query, tables, seq_ids, positions)


def test_decode_agrees_with_the_single_query_kernel():
    """
    The varlen kernel must not disagree with the one it generalises. If these
    two ever diverge, one of them is wrong and the project has two different
    answers for the same question.
    """
    from memforge_ml.kernels.block_attention import paged_attention_blockwise

    pool = make_pool()
    length = 11
    tables = tables_for([length], pool)
    query = make_query(1)

    varlen = paged_attention_varlen(
        query, pool.key_pool, pool.value_pool, 0, tables, [0], [length - 1], BLOCK
    )
    single = paged_attention_blockwise(
        query, pool.key_pool, pool.value_pool, 0, tables[0], length, BLOCK
    )
    assert torch.allclose(varlen, single, atol=ATOL)


# ---------------------------------------------------- prefill-only equivalence

@pytest.mark.parametrize("length", [1, 2, 4, 5, 8, 13, 16, 17])
def test_full_prefill_matches_reference(length):
    """
    A whole prompt as one packed run: positions 0..length-1, which is ordinary
    causal self-attention expressed per token. Lengths straddle the block
    boundary at 4 deliberately -- a partial final block is where an off-by-one
    on `valid` hides.
    """
    pool = make_pool()
    tables = tables_for([length], pool)
    query = make_query(length)
    assert_matches_reference(pool, query, tables, [0] * length, list(range(length)))


@pytest.mark.parametrize("start,size", [(0, 4), (4, 4), (2, 3), (8, 5), (5, 7), (13, 3)])
def test_prefill_chunk_matches_reference(start, size):
    """
    A chunk in the middle of a prompt: positions `start..start+size-1`, seeing
    everything before `start` plus its own causal prefix. This is exactly what
    chunked prefill feeds, and the offset chunks are the interesting ones --
    a chunk starting mid-block tests that causality comes from `positions` and
    not from the packing index.
    """
    pool = make_pool()
    total_context = start + size
    tables = tables_for([total_context], pool)
    query = make_query(size)
    positions = list(range(start, start + size))
    assert_matches_reference(pool, query, tables, [0] * size, positions)


# ------------------------------------------------------- the fused case

@pytest.mark.parametrize("chunk", [1, 3, 4, 7, 12])
def test_prefill_chunk_fused_with_decodes_matches_reference(chunk):
    """
    The case the whole kernel exists for: one sequence contributing a prefill
    chunk while others contribute a single decode token, all in one pass.

    If this is right, prefill and decode can share a forward pass -- which is
    Phase C's second bullet and the reason its stall ratio floors at 2.5x.
    """
    pool = make_pool(num_blocks=96)
    prefill_context = 9 + chunk      # 9 tokens already resident, then the chunk
    decode_lengths = [6, 17, 3]
    tables = tables_for([prefill_context] + decode_lengths, pool)

    seq_ids = [0] * chunk + [1, 2, 3]
    positions = list(range(9, 9 + chunk)) + [n - 1 for n in decode_lengths]
    query = make_query(len(seq_ids))
    assert_matches_reference(pool, query, tables, seq_ids, positions)


def test_interleaved_packing_order_does_not_matter():
    """
    Rows are addressed by `seq_ids`/`positions`, not by their place in the
    packed run, so shuffling the packing must permute the output and change
    nothing else. A kernel that secretly depended on contiguous per-sequence
    runs would pass every test above and fail this one.
    """
    pool = make_pool(num_blocks=96)
    tables = tables_for([12, 6, 20], pool)

    seq_ids = [0, 0, 0, 1, 2]
    positions = [9, 10, 11, 5, 19]
    query = make_query(5, seed=3)
    straight = assert_matches_reference(pool, query, tables, seq_ids, positions)

    order = [3, 0, 4, 2, 1]
    shuffled_ids = [seq_ids[i] for i in order]
    shuffled_pos = [positions[i] for i in order]
    shuffled_q = query[:, :, order, :].contiguous()
    shuffled = assert_matches_reference(
        pool, shuffled_q, tables, shuffled_ids, shuffled_pos
    )

    for new_row, old_row in enumerate(order):
        assert torch.allclose(
            shuffled[:, :, new_row, :], straight[:, :, old_row, :], atol=ATOL
        ), f"row {old_row} changed when the packing order did"


def test_two_sequences_may_share_one_sequence_id_slot():
    """
    Several sequences contributing chunks at once, which is what a step admitting
    two long prompts under a token budget produces.
    """
    pool = make_pool(num_blocks=96)
    tables = tables_for([10, 14], pool)
    seq_ids = [0, 0, 0, 1, 1]
    positions = [7, 8, 9, 12, 13]
    query = make_query(5, seed=5)
    assert_matches_reference(pool, query, tables, seq_ids, positions)


# --------------------------------------------------------------- causality

def test_a_token_cannot_see_later_positions():
    """
    Causality asserted directly rather than inferred from matching a causal
    reference: perturbing a key *after* a query's position must not move that
    query's output at all.

    Bidirectional attention during prefill is a bug this project has already
    had once -- `create_causal_mask` returned None under `allow_is_causal_skip`
    -- and it produces confident wrong text, not a crash.
    """
    pool = make_pool()
    length = 10
    tables = tables_for([length], pool)
    query = make_query(length, seed=7)
    seq_ids, positions = [0] * length, list(range(length))

    before = paged_attention_varlen(
        query, pool.key_pool, pool.value_pool, 0, tables, seq_ids, positions, BLOCK
    )

    # Position 7 lives in logical block 1, slot 3.
    physical = int(tables[0][7 // BLOCK])
    pool.key_pool[0, physical, 7 % BLOCK] += 5.0
    pool.value_pool[0, physical, 7 % BLOCK] += 5.0

    after = paged_attention_varlen(
        query, pool.key_pool, pool.value_pool, 0, tables, seq_ids, positions, BLOCK
    )

    for row in range(7):
        assert torch.equal(before[:, :, row, :], after[:, :, row, :]), (
            f"row {row} saw a change at position 7, which is after it"
        )
    # Non-vacuity: the perturbation must reach the rows that *can* see it, or
    # the equalities above prove nothing.
    for row in range(7, length):
        assert not torch.equal(before[:, :, row, :], after[:, :, row, :]), (
            f"row {row} did not see position 7, which it should"
        )


def test_one_sequence_cannot_see_another():
    """Sequences in the same packed pass must be fully isolated."""
    pool = make_pool(num_blocks=96)
    tables = tables_for([8, 8], pool)
    seq_ids, positions = [0, 1], [7, 7]
    query = make_query(2, seed=11)

    before = paged_attention_varlen(
        query, pool.key_pool, pool.value_pool, 0, tables, seq_ids, positions, BLOCK
    )
    # Disturb everything sequence 1 owns.
    for logical in range(tables[1].numel()):
        pool.key_pool[0, int(tables[1][logical])] += 3.0
        pool.value_pool[0, int(tables[1][logical])] += 3.0
    after = paged_attention_varlen(
        query, pool.key_pool, pool.value_pool, 0, tables, seq_ids, positions, BLOCK
    )

    assert torch.equal(before[:, :, 0, :], after[:, :, 0, :]), "sequence 0 read seq 1"
    assert not torch.equal(before[:, :, 1, :], after[:, :, 1, :]), "seq 1 unaffected?"


def test_unwritten_slots_of_a_partial_block_are_not_read():
    """
    Only `position+1` tokens are valid; the rest of the final block is
    uninitialised. Reading it mixes another request's leftover KV into the
    softmax, which is a silent correctness bug and a cross-request information
    leak at the same time.
    """
    pool = make_pool()
    length = 6  # block 0 full, block 1 holds 2 valid tokens of 4
    tables = tables_for([length], pool)
    query = make_query(length, seed=13)
    seq_ids, positions = [0] * length, list(range(length))

    before = paged_attention_varlen(
        query, pool.key_pool, pool.value_pool, 0, tables, seq_ids, positions, BLOCK
    )
    # Slots 2 and 3 of the final block are past the end of the sequence.
    final = int(tables[0][1])
    pool.key_pool[0, final, 2:] += 100.0
    pool.value_pool[0, final, 2:] += 100.0
    after = paged_attention_varlen(
        query, pool.key_pool, pool.value_pool, 0, tables, seq_ids, positions, BLOCK
    )
    assert torch.equal(before, after), "the kernel read past the end of the sequence"


# ------------------------------------------------- grouped-query attention

@pytest.mark.parametrize("num_heads,kv_heads", [(4, 4), (4, 2), (8, 2), (6, 3)])
def test_grouped_query_attention_matches_reference(num_heads, kv_heads):
    pool = make_pool(kv_heads=kv_heads)
    tables = tables_for([13, 7], pool)
    seq_ids = [0, 0, 0, 1]
    positions = [10, 11, 12, 6]
    query = make_query(4, num_heads=num_heads, seed=17)
    assert_matches_reference(pool, query, tables, seq_ids, positions)


def test_each_layer_reads_its_own_slice_of_the_pool():
    """A layer index that leaked would silently blend layers' KV together."""
    pool = make_pool(layers=3)
    tables = tables_for([9], pool)
    query = make_query(2, seed=19)
    seq_ids, positions = [0, 0], [7, 8]

    outs = [
        assert_matches_reference(pool, query, tables, seq_ids, positions, layer=layer)
        for layer in range(3)
    ]
    for a in range(3):
        for b in range(a + 1, 3):
            assert not torch.allclose(outs[a], outs[b], atol=1e-4), (
                f"layers {a} and {b} gave the same answer"
            )


def test_custom_scale_is_applied():
    pool = make_pool()
    tables = tables_for([8], pool)
    query = make_query(3, seed=23)
    seq_ids, positions = [0] * 3, [5, 6, 7]

    default = paged_attention_varlen(
        query, pool.key_pool, pool.value_pool, 0, tables, seq_ids, positions, BLOCK
    )
    explicit = paged_attention_varlen(
        query, pool.key_pool, pool.value_pool, 0, tables, seq_ids, positions, BLOCK,
        scale=1.0 / math.sqrt(query.shape[3]),
    )
    assert torch.allclose(default, explicit, atol=1e-7)

    other = paged_attention_varlen(
        query, pool.key_pool, pool.value_pool, 0, tables, seq_ids, positions, BLOCK,
        scale=0.5,
    )
    assert not torch.allclose(default, other, atol=1e-4)


# ------------------------------------------------------------- validation

def test_rejects_a_mismatched_metadata_length():
    pool = make_pool()
    tables = tables_for([8], pool)
    query = make_query(3)
    with pytest.raises(ValueError, match="query tokens but"):
        paged_attention_varlen(
            query, pool.key_pool, pool.value_pool, 0, tables, [0, 0], [5, 6, 7], BLOCK
        )


def test_rejects_an_empty_pack():
    pool = make_pool()
    tables = tables_for([8], pool)
    with pytest.raises(ValueError, match="at least one query token"):
        paged_attention_varlen(
            query=torch.zeros(1, 4, 0, 8), key_pool=pool.key_pool,
            value_pool=pool.value_pool, layer_idx=0, block_tables=tables,
            seq_ids=[], positions=[], block_size=BLOCK,
        )


def test_rejects_a_negative_position():
    pool = make_pool()
    tables = tables_for([8], pool)
    with pytest.raises(ValueError, match="negative query position"):
        paged_attention_varlen(
            make_query(2), pool.key_pool, pool.value_pool, 0, tables,
            [0, 0], [3, -1], BLOCK,
        )


def test_rejects_a_seq_id_with_no_block_table():
    pool = make_pool()
    tables = tables_for([8], pool)
    with pytest.raises(IndexError, match="no block table"):
        paged_attention_varlen(
            make_query(2), pool.key_pool, pool.value_pool, 0, tables,
            [0, 3], [3, 3], BLOCK,
        )


def test_rejects_a_position_past_the_block_table():
    """
    Reading past the block table would address an unrelated sequence's block.
    Refused loudly: the alternative is a silent cross-request KV read.
    """
    pool = make_pool()
    tables = tables_for([8], pool)  # 2 blocks, 8 positions
    with pytest.raises(IndexError, match="block table has"):
        paged_attention_varlen(
            make_query(1), pool.key_pool, pool.value_pool, 0, tables,
            [0], [40], BLOCK,
        )


def test_rejects_a_head_count_that_is_not_a_multiple_of_kv_heads():
    pool = make_pool(kv_heads=3)
    tables = tables_for([8], pool)
    with pytest.raises(ValueError, match="not a multiple of"):
        paged_attention_varlen(
            make_query(1, num_heads=4), pool.key_pool, pool.value_pool, 0, tables,
            [0], [7], BLOCK,
        )


# ------------------------------------- the two implementations must agree

@pytest.mark.parametrize(
    "seq_ids,positions,lengths",
    [
        ([0], [0], [1]),
        ([0], [10], [11]),
        ([0, 0, 0], [5, 6, 7], [8]),
        ([0, 1, 2], [5, 16, 2], [6, 17, 3]),
        # The fused case: a chunk beside single decode tokens.
        ([0, 0, 0, 0, 1, 2], [9, 10, 11, 12, 5, 19], [13, 6, 20]),
        # Interleaved packing, to pin that neither implementation depends on
        # contiguous per-sequence runs.
        ([1, 0, 2, 0], [5, 9, 19, 10], [11, 6, 20]),
    ],
)
def test_materialised_matches_block_wise(seq_ids, positions, lengths):
    """
    Two implementations of one function, kept because each wins somewhere: the
    block-wise one bounds working memory and is the shape a GPU kernel takes,
    the materialised one is roughly an order of magnitude faster on CPU. Two
    implementations are only safe while they provably agree, so this is the
    test that makes keeping both defensible.
    """
    from memforge_ml.kernels.varlen_attention import (
        paged_attention_varlen_materialised,
    )

    pool = make_pool(num_blocks=96)
    tables = tables_for(lengths, pool)
    query = make_query(len(seq_ids), seed=29)

    block_wise = paged_attention_varlen(
        query, pool.key_pool, pool.value_pool, 0, tables,
        seq_ids, positions, BLOCK,
    )
    materialised = paged_attention_varlen_materialised(
        query, pool, 0, tables, seq_ids, positions,
    )
    delta = (block_wise - materialised).abs().max()
    assert torch.allclose(block_wise, materialised, atol=ATOL), (
        f"the two varlen implementations disagree by {delta:.3e}"
    )


def test_materialised_also_matches_the_plain_reference():
    """
    Agreement with the block-wise version is not enough on its own -- both
    could share a wrong idea of causality. So the fast path is also checked
    against the independent reference.
    """
    from memforge_ml.kernels.varlen_attention import (
        paged_attention_varlen_materialised,
    )

    pool = make_pool(num_blocks=96)
    tables = tables_for([13, 6, 20], pool)
    seq_ids = [0, 0, 0, 0, 1, 2]
    positions = [9, 10, 11, 12, 5, 19]
    query = make_query(len(seq_ids), seed=31)

    got = paged_attention_varlen_materialised(
        query, pool, 0, tables, seq_ids, positions
    )
    want = reference_varlen_attention(
        query, pool.key_pool, pool.value_pool, 0, tables, seq_ids, positions, BLOCK
    )
    assert torch.allclose(got, want, atol=ATOL)


@pytest.mark.parametrize("num_heads,kv_heads", [(4, 4), (4, 2), (8, 2)])
def test_materialised_grouped_query_attention(num_heads, kv_heads):
    """
    GQA expands along a different axis here than in the block-wise version
    (the gathered context is `(Hkv, ctx, D)`, not `(ctx, Hkv, D)`), so it is
    worth its own case rather than assuming the shared test covers it.
    """
    from memforge_ml.kernels.varlen_attention import (
        paged_attention_varlen_materialised,
    )

    pool = make_pool(kv_heads=kv_heads, num_blocks=96)
    tables = tables_for([13, 7], pool)
    seq_ids, positions = [0, 0, 0, 1], [10, 11, 12, 6]
    query = make_query(4, num_heads=num_heads, seed=37)

    got = paged_attention_varlen_materialised(
        query, pool, 0, tables, seq_ids, positions
    )
    want = reference_varlen_attention(
        query, pool.key_pool, pool.value_pool, 0, tables, seq_ids, positions, BLOCK
    )
    assert torch.allclose(got, want, atol=ATOL)


def test_materialised_rejects_a_position_past_the_block_table():
    from memforge_ml.kernels.varlen_attention import (
        paged_attention_varlen_materialised,
    )

    pool = make_pool()
    tables = tables_for([8], pool)
    with pytest.raises(IndexError, match="block table has"):
        paged_attention_varlen_materialised(
            make_query(1), pool, 0, tables, [0], [40]
        )
