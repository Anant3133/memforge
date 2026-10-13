"""
Tests for PagedCache, the transformers Cache backed by paged physical blocks.

The central guarantee: a model run with a PagedCache must produce exactly the
same logits as the same model run with the stock contiguous DynamicCache. If
that holds, paging is transparent to the model.
"""

import pytest
import torch
from transformers.cache_utils import DynamicCache

from memforge_ml.runtime.kv_pool import ModelKVSpec, PagedKVPool
from memforge_ml.runtime.paged_cache import PagedCache


@pytest.fixture
def pool(tiny_model):
    spec = ModelKVSpec.from_config(tiny_model.config, torch.float32, "cpu")
    return PagedKVPool(spec, num_blocks=32, block_size=4)


def test_paged_cache_matches_dynamic_cache_exactly(tiny_model, pool, prompt_ids):
    """Prefill plus several decode steps must be bit-identical to stock caching."""
    ids = torch.tensor([prompt_ids], dtype=torch.long)

    reference_cache = DynamicCache()
    with torch.no_grad():
        reference = tiny_model(input_ids=ids, past_key_values=reference_cache, use_cache=True)

    # A scattered, out-of-order block table is the interesting case.
    block_table = [9, 3, 17, 1, 25, 8]
    paged_cache = PagedCache(pool, block_table)
    paged_cache.begin_step(block_table)
    with torch.no_grad():
        paged = tiny_model(input_ids=ids, past_key_values=paged_cache, use_cache=True)

    assert torch.equal(reference.logits, paged.logits)

    # Now decode step by step through both caches.
    next_token = int(reference.logits[0, -1].argmax())
    for _ in range(6):
        step = torch.tensor([[next_token]], dtype=torch.long)

        with torch.no_grad():
            reference = tiny_model(
                input_ids=step, past_key_values=reference_cache, use_cache=True
            )
        paged_cache.begin_step(block_table)
        with torch.no_grad():
            paged = tiny_model(input_ids=step, past_key_values=paged_cache, use_cache=True)

        assert torch.equal(reference.logits, paged.logits)
        next_token = int(reference.logits[0, -1].argmax())


def test_position_tracks_tokens_written(tiny_model, pool, prompt_ids):
    block_table = list(range(6))
    cache = PagedCache(pool, block_table)
    cache.begin_step(block_table)

    assert cache.position == 0
    with torch.no_grad():
        tiny_model(
            input_ids=torch.tensor([prompt_ids]), past_key_values=cache, use_cache=True
        )
    assert cache.position == len(prompt_ids)

    cache.begin_step(block_table)
    with torch.no_grad():
        tiny_model(input_ids=torch.tensor([[5]]), past_key_values=cache, use_cache=True)
    assert cache.position == len(prompt_ids) + 1


def test_all_layers_stay_in_lockstep(tiny_model, pool, prompt_ids):
    block_table = list(range(6))
    cache = PagedCache(pool, block_table)
    cache.begin_step(block_table)
    with torch.no_grad():
        tiny_model(input_ids=torch.tensor([prompt_ids]), past_key_values=cache, use_cache=True)

    positions = {layer.position for layer in cache.layers}
    assert positions == {len(prompt_ids)}


def test_rollback_rewinds_without_touching_the_block_table(tiny_model, pool, prompt_ids):
    """Rejecting speculative tokens must be a cursor move, not a reallocation."""
    block_table = list(range(8))
    cache = PagedCache(pool, block_table)
    cache.begin_step(block_table)
    with torch.no_grad():
        tiny_model(input_ids=torch.tensor([prompt_ids]), past_key_values=cache, use_cache=True)

    # Speculate 4 tokens ahead.
    cache.begin_step(block_table)
    with torch.no_grad():
        tiny_model(
            input_ids=torch.tensor([[3, 4, 5, 6]]), past_key_values=cache, use_cache=True
        )
    assert cache.position == len(prompt_ids) + 4

    cache.rollback(3)

    assert cache.position == len(prompt_ids) + 1
    assert cache.block_table == block_table  # blocks still owned, nothing freed


def test_rollback_then_replay_matches_a_clean_run(tiny_model, pool, prompt_ids):
    """
    After rolling back a rejected speculative tail, continuing from the rollback
    point must give the same logits as if those tokens had never been written.
    """
    block_table = list(range(8))

    # Path A: write a speculative tail, roll it back, then decode the real token.
    cache_a = PagedCache(pool, block_table)
    cache_a.begin_step(block_table)
    with torch.no_grad():
        tiny_model(input_ids=torch.tensor([prompt_ids]), past_key_values=cache_a, use_cache=True)
    cache_a.begin_step(block_table)
    with torch.no_grad():
        tiny_model(input_ids=torch.tensor([[31, 32, 33]]), past_key_values=cache_a, use_cache=True)
    cache_a.rollback(3)
    cache_a.begin_step(block_table)
    with torch.no_grad():
        out_a = tiny_model(input_ids=torch.tensor([[99]]), past_key_values=cache_a, use_cache=True)

    # Path B: a fresh cache in different blocks that only ever sees the real token.
    other_table = list(range(10, 18))
    cache_b = PagedCache(pool, other_table)
    cache_b.begin_step(other_table)
    with torch.no_grad():
        tiny_model(input_ids=torch.tensor([prompt_ids]), past_key_values=cache_b, use_cache=True)
    cache_b.begin_step(other_table)
    with torch.no_grad():
        out_b = tiny_model(input_ids=torch.tensor([[99]]), past_key_values=cache_b, use_cache=True)

    assert torch.equal(out_a.logits, out_b.logits)


def test_reset_returns_the_cache_to_empty(tiny_model, pool, prompt_ids):
    block_table = list(range(6))
    cache = PagedCache(pool, block_table)
    cache.begin_step(block_table)
    with torch.no_grad():
        tiny_model(input_ids=torch.tensor([prompt_ids]), past_key_values=cache, use_cache=True)

    cache.reset()

    assert cache.position == 0
    assert all(layer.position == 0 for layer in cache.layers)


def test_growing_the_block_table_is_picked_up(tiny_model, pool, prompt_ids):
    """The engine grows block tables mid-sequence; the cache must follow."""
    block_table = [0, 1]
    cache = PagedCache(pool, block_table)
    cache.begin_step(block_table)
    with torch.no_grad():
        tiny_model(input_ids=torch.tensor([prompt_ids]), past_key_values=cache, use_cache=True)

    # Pool block_size is 4, so 8 prompt tokens exactly fill 2 blocks; the next
    # token needs a third.
    grown = block_table + [2]
    cache.begin_step(grown)
    with torch.no_grad():
        tiny_model(input_ids=torch.tensor([[5]]), past_key_values=cache, use_cache=True)

    assert cache.block_table == grown
    assert cache.position == len(prompt_ids) + 1


def test_rejects_batched_input(tiny_model, pool):
    cache = PagedCache(pool, list(range(8)))
    cache.begin_step(list(range(8)))
    with pytest.raises(ValueError, match="single-sequence"):
        with torch.no_grad():
            tiny_model(
                input_ids=torch.tensor([[1, 2, 3], [4, 5, 6]]),
                past_key_values=cache,
                use_cache=True,
            )
