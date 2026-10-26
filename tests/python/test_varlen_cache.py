"""
Unit tests for the packed multi-sequence cache (Phase C).

`test_fused_prefill.py` exercises this through a real engine step, which is the
test that matters for behaviour. These cover the parts an engine step cannot
reach: the addressing arithmetic in isolation, and the guards that exist to turn
a caller's mistake into an error instead of a silent cross-sequence write.

The slot-collision guard is the important one. Two packed tokens addressing the
same pool slot would make the pass's result depend on scatter order and would
mean one sequence overwrote another's KV -- which is invisible from the output
and is exactly the class of bug this project treats as the dangerous one.
"""

from __future__ import annotations

import pytest
import torch

from memforge_ml.runtime.kv_pool import ModelKVSpec, PagedKVPool
from memforge_ml.runtime.varlen_cache import VarlenPackedCache

BLOCK = 4
LAYERS = 2


def make_pool(num_blocks=32, kv_heads=2, head_dim=8):
    spec = ModelKVSpec(
        num_layers=LAYERS, num_kv_heads=kv_heads, head_dim=head_dim,
        dtype=torch.float32, device="cpu",
    )
    pool = PagedKVPool(spec, num_blocks=num_blocks, block_size=BLOCK)
    pool.key_pool.zero_()
    pool.value_pool.zero_()
    return pool


def table(ids):
    return PagedKVPool.block_table_tensor(ids, "cpu")


# ------------------------------------------------------------- addressing

def test_write_index_addresses_each_token_through_its_own_table():
    """
    The whole difference from the single-sequence cache: token *i*'s slot comes
    from sequence `seq_ids[i]`'s block table, not from a shared one.
    """
    pool = make_pool()
    tables = [table([5, 2]), table([7])]
    # seq 0 positions 4 and 5 -> its logical block 1 = physical 2, slots 0 and 1
    # seq 1 position 2        -> its logical block 0 = physical 7, slot 2
    cache = VarlenPackedCache(pool, LAYERS, tables, [0, 0, 1], [4, 5, 2])
    assert cache.write_index.tolist() == [
        2 * BLOCK + 0,
        2 * BLOCK + 1,
        7 * BLOCK + 2,
    ]


def test_write_index_is_independent_of_packing_order():
    pool = make_pool()
    tables = [table([5, 2]), table([7])]
    straight = VarlenPackedCache(pool, LAYERS, tables, [0, 0, 1], [4, 5, 2])
    shuffled = VarlenPackedCache(pool, LAYERS, tables, [1, 0, 0], [2, 5, 4])
    assert sorted(straight.write_index.tolist()) == sorted(
        shuffled.write_index.tolist()
    )


def test_update_scatters_into_the_addressed_slots():
    """One layer's write must land exactly where the index says, and nowhere else."""
    pool = make_pool()
    tables = [table([5, 2]), table([7])]
    cache = VarlenPackedCache(pool, LAYERS, tables, [0, 0, 1], [4, 5, 2])

    keys = torch.arange(3 * 2 * 8, dtype=torch.float32).reshape(1, 2, 3, 8)
    values = keys * -1.0
    returned_k, returned_v = cache.layers[0].update(keys, values, 0)

    # Lazy: the states come straight back, because there is no single context
    # to return for tokens from different sequences.
    assert torch.equal(returned_k, keys)
    assert torch.equal(returned_v, values)

    # The three addressed slots hold the new KV.
    for row, (physical, slot) in enumerate([(2, 0), (2, 1), (7, 2)]):
        assert torch.equal(
            pool.key_pool[0, physical, slot], keys[0, :, row, :]
        ), f"row {row} did not land in block {physical} slot {slot}"
        assert torch.equal(pool.value_pool[0, physical, slot], values[0, :, row, :])

    # And layer 1 was untouched, so the layer index is honoured.
    assert pool.key_pool[1].abs().sum() == 0


def test_each_layer_writes_its_own_slice():
    pool = make_pool()
    tables = [table([3])]
    cache = VarlenPackedCache(pool, LAYERS, tables, [0], [1])
    for layer_idx in range(LAYERS):
        keys = torch.full((1, 2, 1, 8), float(layer_idx + 1))
        cache.layers[layer_idx].update(keys, keys, layer_idx)
    assert pool.key_pool[0, 3, 1].unique().tolist() == [1.0]
    assert pool.key_pool[1, 3, 1].unique().tolist() == [2.0]


# --------------------------------------------------------------- guards

def test_two_tokens_cannot_claim_the_same_slot():
    """
    A collision means one sequence silently overwrote another's KV, and the
    result would depend on scatter order. Refused at construction.
    """
    pool = make_pool()
    tables = [table([5]), table([5])]  # same physical block, by mistake
    with pytest.raises(ValueError, match="same pool slot"):
        VarlenPackedCache(pool, LAYERS, tables, [0, 1], [1, 1])


def test_a_sequence_cannot_write_the_same_position_twice():
    pool = make_pool()
    tables = [table([5])]
    with pytest.raises(ValueError, match="same pool slot"):
        VarlenPackedCache(pool, LAYERS, tables, [0, 0], [2, 2])


def test_rejects_a_position_past_the_block_table():
    pool = make_pool()
    tables = [table([5])]  # one block, positions 0-3
    with pytest.raises(IndexError, match="block table of"):
        VarlenPackedCache(pool, LAYERS, tables, [0], [9])


def test_rejects_an_unknown_sequence_id():
    pool = make_pool()
    with pytest.raises(IndexError, match="no block table"):
        VarlenPackedCache(pool, LAYERS, [table([5])], [3], [1])


def test_rejects_a_negative_position():
    pool = make_pool()
    with pytest.raises(ValueError, match="negative position"):
        VarlenPackedCache(pool, LAYERS, [table([5])], [0], [-1])


def test_rejects_mismatched_metadata():
    pool = make_pool()
    with pytest.raises(ValueError, match="seq_ids but"):
        VarlenPackedCache(pool, LAYERS, [table([5])], [0, 0], [1])


def test_rejects_an_empty_pass():
    pool = make_pool()
    with pytest.raises(ValueError, match="at least one token"):
        VarlenPackedCache(pool, LAYERS, [table([5])], [], [])


def test_update_rejects_a_real_batch_dimension():
    """
    This cache is packed, not batched. A batch dimension above 1 means the
    caller built the wrong layout, and padding it would waste exactly the
    compute fusion exists to save.
    """
    pool = make_pool()
    cache = VarlenPackedCache(pool, LAYERS, [table([5])], [0], [1])
    keys = torch.zeros(2, 2, 1, 8)
    with pytest.raises(ValueError, match="one packed row"):
        cache.layers[0].update(keys, keys, 0)


def test_update_rejects_the_wrong_token_count():
    pool = make_pool()
    cache = VarlenPackedCache(pool, LAYERS, [table([5])], [0, 0], [0, 1])
    keys = torch.zeros(1, 2, 3, 8)  # 3 states for a 2-token pass
    with pytest.raises(ValueError, match="for 2 packed tokens"):
        cache.layers[0].update(keys, keys, 0)


def test_crop_is_refused_rather_than_silently_ignored():
    """
    There is no per-sequence cursor here to roll back, so cropping cannot mean
    what a caller would expect. Undoing a fused step means not advancing the
    owning sequences' own cursors, and a no-op `crop` would hide that.
    """
    pool = make_pool()
    cache = VarlenPackedCache(pool, LAYERS, [table([5])], [0], [1])
    with pytest.raises(NotImplementedError, match="no per-sequence cursor"):
        cache.layers[0].crop(1)


# ------------------------------------------------------- cache interface

def test_reports_no_sequence_length():
    """
    A packed run has no single length. Reporting 0 stops transformers offsetting
    positions itself; the engine passes absolute `position_ids` instead.
    """
    pool = make_pool()
    cache = VarlenPackedCache(pool, LAYERS, [table([5])], [0, 0], [0, 1])
    assert cache.get_seq_length() == 0
    assert cache.layers[0].get_seq_length() == 0
    # Any mask transformers builds covers the packed run, not the longest
    # context in it.
    assert cache.layers[0].get_mask_sizes(2) == (2, 0)
    assert len(cache) == LAYERS
