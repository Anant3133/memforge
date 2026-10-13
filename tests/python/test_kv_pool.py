"""Tests for the paged KV physical block pool."""

import pytest
import torch

from memforge_ml.runtime.kv_pool import ModelKVSpec, PagedKVPool


def make_pool(num_blocks=8, block_size=4, num_layers=2, num_kv_heads=3, head_dim=4):
    spec = ModelKVSpec(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=torch.float32,
        device="cpu",
    )
    return PagedKVPool(spec, num_blocks=num_blocks, block_size=block_size)


def test_slot_indices_translate_logical_positions_to_physical_blocks():
    pool = make_pool(block_size=4)
    block_table = PagedKVPool.block_table_tensor([5, 2, 7], "cpu")

    physical, slots = pool.slot_indices(block_table, 0, 10)

    # Positions 0-3 -> block 5, 4-7 -> block 2, 8-9 -> block 7.
    assert physical.tolist() == [5, 5, 5, 5, 2, 2, 2, 2, 7, 7]
    assert slots.tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]


def test_slot_indices_handles_mid_sequence_range():
    pool = make_pool(block_size=4)
    block_table = PagedKVPool.block_table_tensor([5, 2, 7], "cpu")

    physical, slots = pool.slot_indices(block_table, 6, 9)

    assert physical.tolist() == [2, 2, 7]
    assert slots.tolist() == [2, 3, 0]


def test_slot_indices_rejects_positions_past_the_block_table():
    pool = make_pool(block_size=4)
    block_table = PagedKVPool.block_table_tensor([5, 2], "cpu")

    with pytest.raises(IndexError, match="outside a block table"):
        pool.slot_indices(block_table, 0, 12)  # needs 3 blocks, table has 2


def test_write_then_gather_round_trips_through_scattered_blocks():
    pool = make_pool(num_blocks=8, block_size=4, num_kv_heads=3, head_dim=4)
    # Deliberately out of order and non-contiguous.
    block_table = PagedKVPool.block_table_tensor([6, 1, 4], "cpu")
    physical, slots = pool.slot_indices(block_table, 0, 10)

    keys = torch.randn(1, 3, 10, 4)
    values = torch.randn(1, 3, 10, 4)
    pool.write(1, physical, slots, keys, values)

    got_k, got_v = pool.gather(1, physical, slots)

    assert got_k.shape == keys.shape
    assert torch.equal(got_k, keys)
    assert torch.equal(got_v, values)


def test_layers_are_isolated_from_each_other():
    pool = make_pool(num_layers=3)
    block_table = PagedKVPool.block_table_tensor([0, 1], "cpu")
    physical, slots = pool.slot_indices(block_table, 0, 5)

    pool.write(1, physical, slots, torch.ones(1, 3, 5, 4), torch.ones(1, 3, 5, 4))

    assert pool.key_pool[1].abs().sum() > 0
    assert pool.key_pool[0].abs().sum() == 0
    assert pool.key_pool[2].abs().sum() == 0


def test_sequences_sharing_a_pool_do_not_corrupt_each_other():
    pool = make_pool(num_blocks=8, block_size=4)
    table_a = PagedKVPool.block_table_tensor([0, 1], "cpu")
    table_b = PagedKVPool.block_table_tensor([2, 3], "cpu")

    phys_a, slots_a = pool.slot_indices(table_a, 0, 8)
    phys_b, slots_b = pool.slot_indices(table_b, 0, 8)

    keys_a = torch.full((1, 3, 8, 4), 1.0)
    keys_b = torch.full((1, 3, 8, 4), 2.0)
    pool.write(0, phys_a, slots_a, keys_a, keys_a)
    pool.write(0, phys_b, slots_b, keys_b, keys_b)

    got_a, _ = pool.gather(0, phys_a, slots_a)
    got_b, _ = pool.gather(0, phys_b, slots_b)

    assert torch.equal(got_a, keys_a)
    assert torch.equal(got_b, keys_b)


def test_bytes_per_token_accounts_for_every_layer_and_both_tensors():
    spec = ModelKVSpec(
        num_layers=4, num_kv_heads=2, head_dim=8, dtype=torch.float32, device="cpu"
    )
    # 2 (K+V) * 4 layers * 2 heads * 8 dims * 4 bytes
    assert spec.bytes_per_token == 2 * 4 * 2 * 8 * 4


def test_blocks_for_budget_stays_within_the_memory_budget():
    spec = ModelKVSpec(
        num_layers=4, num_kv_heads=2, head_dim=8, dtype=torch.float32, device="cpu"
    )
    budget_mb = 1.0

    num_blocks = PagedKVPool.blocks_for_budget(spec, block_size=16, budget_mb=budget_mb)
    pool = PagedKVPool(spec, num_blocks=num_blocks, block_size=16)

    assert num_blocks >= 1
    assert pool.total_bytes <= budget_mb * 1024 * 1024
    # And it should be close to the budget, not wildly under it.
    assert pool.total_bytes + pool.bytes_per_block > budget_mb * 1024 * 1024


def test_pool_reports_its_capacity_in_tokens():
    pool = make_pool(num_blocks=10, block_size=16)
    assert pool.capacity_tokens == 160
    assert pool.describe()["capacity_tokens"] == 160
