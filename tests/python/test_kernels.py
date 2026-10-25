"""
Differential tests for the Phase A read/write kernels.

Every test here compares an optimised path against the straightforward
reference in `PagedKVPool`, and compares with `torch.equal` rather than
`allclose`. That is deliberate: these kernels only reorder memory accesses, so
they must be *bit*-identical, not merely close. A kernel that is almost right
produces fluent, confident, wrong text instead of a crash, which no
approximate assertion would catch.
"""

from __future__ import annotations

import pytest
import torch

from memforge_ml.kernels.batched_decode import BatchedDecodeBuffers, assemble_batch
from memforge_ml.kernels.paged_gather import flat_slot_index, gather_kv, scatter_kv
from memforge_ml.runtime.kv_pool import ModelKVSpec, PagedKVPool

BLOCK = 4


def make_pool(num_blocks: int = 16, layers: int = 3, heads: int = 2, dim: int = 8):
    spec = ModelKVSpec(
        num_layers=layers, num_kv_heads=heads, head_dim=dim,
        dtype=torch.float32, device="cpu",
    )
    pool = PagedKVPool(spec, num_blocks=num_blocks, block_size=BLOCK)
    torch.manual_seed(0)
    pool.key_pool.normal_()
    pool.value_pool.normal_()
    return pool


# ------------------------------------------------------------- flat indexing

def test_flat_slot_index_matches_manual_arithmetic():
    physical = torch.tensor([3, 1, 0, 2])
    slots = torch.tensor([0, 2, 3, 1])
    flat = flat_slot_index(physical, slots, BLOCK)
    assert flat.tolist() == [12, 6, 3, 9]
    # index_select needs a contiguous index or it copies on every call.
    assert flat.is_contiguous()


def test_flat_indices_agrees_with_slot_indices():
    """The two index forms must address the same elements."""
    pool = make_pool()
    table = PagedKVPool.block_table_tensor([5, 2, 9, 1], "cpu")
    physical, slots = pool.slot_indices(table, 0, 13)
    flat = pool.flat_indices(table, 0, 13)
    assert torch.equal(flat, physical * BLOCK + slots)


# --------------------------------------------------------------- gather path

@pytest.mark.parametrize("token_range", [(0, 1), (0, 4), (0, 13), (5, 13), (12, 13)])
def test_gather_flat_is_bit_identical_to_gather(token_range):
    pool = make_pool()
    table = PagedKVPool.block_table_tensor([7, 3, 11, 0], "cpu")
    start, end = token_range
    physical, slots = pool.slot_indices(table, start, end)
    flat = pool.flat_indices(table, start, end)

    for layer in range(pool.spec.num_layers):
        ref_k, ref_v = pool.gather(layer, physical, slots)
        got_k, got_v = pool.gather_flat(layer, flat)
        assert torch.equal(ref_k, got_k), f"keys differ, layer {layer}"
        assert torch.equal(ref_v, got_v), f"values differ, layer {layer}"
        assert got_k.shape == (1, pool.spec.num_kv_heads, end - start, pool.spec.head_dim)


def test_gather_handles_a_scattered_block_table():
    """
    Paging's whole point: a sequence's blocks are not contiguous or ordered.

    A gather that quietly assumed ascending block ids would pass a tidy test and
    fail here.
    """
    pool = make_pool(num_blocks=20)
    table = PagedKVPool.block_table_tensor([19, 0, 13, 4, 7], "cpu")
    physical, slots = pool.slot_indices(table, 0, 20)
    flat = pool.flat_indices(table, 0, 20)
    ref_k, ref_v = pool.gather(0, physical, slots)
    got_k, got_v = pool.gather_flat(0, flat)
    assert torch.equal(ref_k, got_k)
    assert torch.equal(ref_v, got_v)


def test_gather_kv_rejects_a_block_size_mismatch():
    pool = make_pool()
    flat = torch.tensor([0, 1, 2])
    with pytest.raises(AssertionError, match="block size"):
        gather_kv(pool.key_pool, pool.value_pool, 0, flat, BLOCK + 1)


# -------------------------------------------------------------- scatter path

def test_write_flat_round_trips_through_gather():
    pool = make_pool()
    table = PagedKVPool.block_table_tensor([2, 6, 1], "cpu")
    flat = pool.flat_indices(table, 0, 10)
    physical, slots = pool.slot_indices(table, 0, 10)

    keys = torch.randn(1, pool.spec.num_kv_heads, 10, pool.spec.head_dim)
    values = torch.randn(1, pool.spec.num_kv_heads, 10, pool.spec.head_dim)
    pool.write_flat(1, flat, keys, values)

    got_k, got_v = pool.gather(1, physical, slots)
    assert torch.equal(got_k, keys)
    assert torch.equal(got_v, values)


def test_write_flat_matches_write():
    """The optimised scatter must land bytes exactly where the reference does."""
    pool_a = make_pool()
    pool_b = make_pool()
    table = PagedKVPool.block_table_tensor([4, 0, 9], "cpu")
    physical, slots = pool_a.slot_indices(table, 2, 11)
    flat = pool_a.flat_indices(table, 2, 11)

    keys = torch.randn(1, pool_a.spec.num_kv_heads, 9, pool_a.spec.head_dim)
    values = torch.randn(1, pool_a.spec.num_kv_heads, 9, pool_a.spec.head_dim)

    pool_a.write(0, physical, slots, keys, values)
    pool_b.write_flat(0, flat, keys, values)
    assert torch.equal(pool_a.key_pool, pool_b.key_pool)
    assert torch.equal(pool_a.value_pool, pool_b.value_pool)


def test_write_flat_touches_only_the_addressed_slots():
    """A scatter that overwrote neighbouring slots would corrupt other sequences."""
    pool = make_pool()
    before_k = pool.key_pool.clone()
    table = PagedKVPool.block_table_tensor([3], "cpu")
    flat = pool.flat_indices(table, 0, 2)  # only slots 0 and 1 of block 3

    keys = torch.randn(1, pool.spec.num_kv_heads, 2, pool.spec.head_dim)
    pool.write_flat(0, flat, keys, keys)

    untouched = torch.ones_like(before_k, dtype=torch.bool)
    untouched[0, 3, 0] = False
    untouched[0, 3, 1] = False
    assert torch.equal(pool.key_pool[untouched], before_k[untouched])


# ----------------------------------------------------- batched decode buffers

def test_buffers_round_up_and_reuse_shapes():
    """
    Contexts grow one token per step, so rounding up is what stops the engine
    allocating a fresh buffer pair on every single step.
    """
    buffers = BatchedDecodeBuffers(num_layers=2, pad_to=64)
    assert buffers.padded_len(1) == 64
    assert buffers.padded_len(64) == 64
    assert buffers.padded_len(65) == 128

    # Every length in 65..128 shares the 128 bucket, so a context growing a
    # token at a time reuses one buffer pair instead of allocating per step.
    for length in (100, 101, 120, 128):
        buffers.acquire(2, length, 2, 8, torch.float32, "cpu")
    assert buffers.num_cached_shapes == 1

    # Crossing into the next bucket allocates once more, and no more than once.
    for length in (129, 150, 192):
        buffers.acquire(2, length, 2, 8, torch.float32, "cpu")
    assert buffers.num_cached_shapes == 2

    # A different batch size is a different buffer: the first dimension changed.
    buffers.acquire(4, 100, 2, 8, torch.float32, "cpu")
    assert buffers.num_cached_shapes == 3

    buffers.clear()
    assert buffers.num_cached_shapes == 0


def test_buffers_are_zeroed_on_reuse():
    """
    The padded region is read by the model and masked out multiplicatively, so
    stale non-finite values there would poison the softmax even though they do
    not "count".
    """
    buffers = BatchedDecodeBuffers(num_layers=1, pad_to=8)
    keys, _, _ = buffers.acquire(1, 8, 2, 4, torch.float32, "cpu")
    keys.fill_(float("nan"))

    keys_again, values_again, _ = buffers.acquire(1, 8, 2, 4, torch.float32, "cpu")
    assert not torch.isnan(keys_again).any()
    assert torch.all(keys_again == 0)
    assert torch.all(values_again == 0)


def test_assemble_batch_matches_the_pad_and_cat_reference():
    """
    The behaviour tier 1b replaces: gather, F.pad to the batch maximum, cat.

    Compared over the real-token region only, because the two paths pad to
    different widths by design -- the reference pads to the exact maximum, the
    kernel to a reusable bucket.
    """
    import torch.nn.functional as F

    pool = make_pool(num_blocks=40, layers=3)
    batch_ctx = [10, 7, 13]
    tables = [
        PagedKVPool.block_table_tensor([0, 1, 2, 3], "cpu"),
        PagedKVPool.block_table_tensor([10, 11, 12], "cpu"),
        PagedKVPool.block_table_tensor([20, 21, 22, 23], "cpu"),
    ]
    max_len = max(batch_ctx)

    reference = []
    for layer in range(pool.spec.num_layers):
        ks, vs = [], []
        for table, ctx in zip(tables, batch_ctx):
            physical, slots = pool.slot_indices(table, 0, ctx)
            k, v = pool.gather(layer, physical, slots)
            pad = max_len - ctx
            if pad:
                k = F.pad(k, (0, 0, pad, 0))
                v = F.pad(v, (0, 0, pad, 0))
            ks.append(k)
            vs.append(v)
        reference.append((torch.cat(ks, dim=0), torch.cat(vs, dim=0)))

    buffers = BatchedDecodeBuffers(pool.spec.num_layers, pad_to=8)
    got, mask, padded = assemble_batch(pool, buffers, tables, batch_ctx)

    assert padded >= max_len
    for layer in range(pool.spec.num_layers):
        for row, ctx in enumerate(batch_ctx):
            ref_k = reference[layer][0][row, :, max_len - ctx:, :]
            ref_v = reference[layer][1][row, :, max_len - ctx:, :]
            got_k = got[layer][0][row, :, padded - ctx:, :]
            got_v = got[layer][1][row, :, padded - ctx:, :]
            assert torch.equal(ref_k, got_k), f"keys differ, layer {layer} row {row}"
            assert torch.equal(ref_v, got_v), f"values differ, layer {layer} row {row}"

    # The mask must mark exactly the real tokens, left-aligned to the row end.
    for row, ctx in enumerate(batch_ctx):
        assert mask[row, padded - ctx:].sum().item() == ctx
        assert mask[row, : padded - ctx].sum().item() == 0


def test_assemble_batch_pads_region_is_zero():
    pool = make_pool(num_blocks=20)
    tables = [PagedKVPool.block_table_tensor([0, 1], "cpu"),
              PagedKVPool.block_table_tensor([10, 11], "cpu")]
    buffers = BatchedDecodeBuffers(pool.spec.num_layers, pad_to=16)
    got, _, padded = assemble_batch(pool, buffers, tables, [5, 3])

    for layer in range(pool.spec.num_layers):
        assert torch.all(got[layer][0][0, :, : padded - 5, :] == 0)
        assert torch.all(got[layer][0][1, :, : padded - 3, :] == 0)


def test_assemble_batch_rejects_mismatched_inputs():
    pool = make_pool()
    buffers = BatchedDecodeBuffers(pool.spec.num_layers)
    table = PagedKVPool.block_table_tensor([0, 1], "cpu")

    with pytest.raises(ValueError, match="at least one sequence"):
        assemble_batch(pool, buffers, [], [])
    with pytest.raises(ValueError, match="block tables"):
        assemble_batch(pool, buffers, [table], [4, 4])


def test_scatter_kv_is_the_exact_inverse_of_gather_kv():
    """
    Direct coverage of the kernel pair, rather than only through the pool wrappers.

    Gather, perturb, scatter, gather again: the second read must return exactly
    what was written. A scatter that transposed differently from the gather would
    round-trip wrong and nothing else here would notice.
    """
    pool = make_pool(num_blocks=12)
    table = PagedKVPool.block_table_tensor([9, 4, 1], "cpu")
    physical, slots = pool.slot_indices(table, 0, 11)
    flat = flat_slot_index(physical, slots, BLOCK)

    original_k, original_v = gather_kv(
        pool.key_pool, pool.value_pool, 0, flat, BLOCK
    )
    new_k = original_k + 1.5
    new_v = original_v - 0.25

    scatter_kv(pool.key_pool, pool.value_pool, 0, flat, new_k, new_v, BLOCK)
    got_k, got_v = gather_kv(pool.key_pool, pool.value_pool, 0, flat, BLOCK)

    assert torch.equal(got_k, new_k)
    assert torch.equal(got_v, new_v)
    # Other layers must be untouched: the layer index has to be honoured.
    assert not torch.equal(pool.key_pool[1], pool.key_pool[0])
