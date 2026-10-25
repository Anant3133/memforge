"""
Tests for the append-only decode mirror (Phase A, tier 2).

The mirror is a *cache of the pool*, and the pool is the source of truth. So the
tests fall into two groups: that the mirror returns exactly what a pool read
would have returned, and that every event which can change the pool underneath
it causes a re-read rather than a stale hit.

Comparisons use `torch.equal`, not `allclose`. The mirror only reorders memory
accesses, so it must be bit-identical; a nearly-right KV cache produces fluent,
confident, wrong text rather than a crash.
"""

from __future__ import annotations

import pytest
import torch

from memforge_ml.runtime.decode_view_cache import DecodeViewCache, capacity_for
from memforge_ml.runtime.kv_pool import ModelKVSpec, PagedKVPool
from memforge_ml.runtime.paged_cache import PagedCache

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


# ------------------------------------------------------------------- sizing

def test_capacity_is_sized_from_the_request_not_the_model_limit():
    """
    The gap between `prompt + max_new_tokens` and `max_model_len` is where the
    memory cost lives: at SmolLM2-360M geometry a 1024-token mirror is ~84 MB
    per sequence, so sizing from the model limit would waste most of it.
    """
    assert capacity_for(100, 20) == 121        # +1 slack for a speculative bonus token
    assert capacity_for(0, 0) == 1             # never zero-sized
    assert capacity_for(1, 1, slack=0) == 2


def test_rejects_a_non_positive_capacity():
    spec = make_pool().spec
    with pytest.raises(ValueError, match="capacity_tokens must be positive"):
        DecodeViewCache(spec, 0)


def test_reports_its_own_cost():
    """The memory cost has to be visible, since it is the reason this is opt-in."""
    pool = make_pool(layers=4, heads=2, dim=8)
    mirror = DecodeViewCache(pool.spec, capacity_tokens=100)
    described = mirror.describe()
    # 2 tensors x 4 layers x 2 heads x 100 tokens x 8 dims x 4 bytes
    assert described["bytes"] == 2 * 4 * 2 * 100 * 8 * 4
    assert described["capacity_tokens"] == 100
    assert described["filled_tokens"] == 0


# ------------------------------------------------------------ append and read

def test_append_then_view_round_trips():
    pool = make_pool()
    mirror = DecodeViewCache(pool.spec, capacity_tokens=12)
    heads, dim = pool.spec.num_kv_heads, pool.spec.head_dim

    written = []
    for step in range(5):
        k = torch.randn(1, heads, 1, dim)
        v = torch.randn(1, heads, 1, dim)
        written.append((k, v))
        for layer in range(pool.spec.num_layers):
            mirror.append(layer, k, v, step)

    assert mirror.filled == 5
    for layer in range(pool.spec.num_layers):
        got_k, got_v = mirror.view(layer, 5)
        assert got_k.shape == (1, heads, 5, dim)
        for step, (k, v) in enumerate(written):
            assert torch.equal(got_k[:, :, step : step + 1, :], k)
            assert torch.equal(got_v[:, :, step : step + 1, :], v)


def test_view_aliases_the_buffer_rather_than_copying():
    """The entire point of the mirror: a read must not allocate."""
    pool = make_pool()
    mirror = DecodeViewCache(pool.spec, capacity_tokens=8)
    got_k, got_v = mirror.view(0, 4)

    assert got_k.data_ptr() == mirror.keys[0].data_ptr()
    assert got_v.data_ptr() == mirror.values[0].data_ptr()
    # Writing through the view is visible in the buffer, i.e. it aliases.
    got_k.fill_(3.5)
    assert torch.all(mirror.keys[0, :, :, :4, :] == 3.5)


def test_filled_advances_only_once_every_layer_has_written():
    """
    A token is valid only when all layers hold it.

    Every layer writes the same position range, so a counter advanced per layer
    would over-report by `num_layers` and expose half-written tokens.
    """
    pool = make_pool(layers=3)
    mirror = DecodeViewCache(pool.spec, capacity_tokens=4)
    k = torch.randn(1, pool.spec.num_kv_heads, 1, pool.spec.head_dim)

    mirror.append(0, k, k, 0)
    assert mirror.filled == 0, "token visible before every layer wrote it"
    mirror.append(1, k, k, 0)
    assert mirror.filled == 0
    mirror.append(2, k, k, 0)
    assert mirror.filled == 1


def test_unwritten_positions_read_as_zero():
    """
    Predictable rather than arbitrary: a bookkeeping bug then shows up as
    wrong-but-finite output instead of a NaN storm several layers later.
    """
    pool = make_pool()
    mirror = DecodeViewCache(pool.spec, capacity_tokens=6)
    got_k, got_v = mirror.view(0, 6)
    assert torch.all(got_k == 0)
    assert torch.all(got_v == 0)


# ------------------------------------------------------------------- seeding

def test_seed_matches_a_pool_read():
    """
    After a prefix-cache hit the leading tokens were written by a *different*
    sequence, so the mirror has never seen them and must read the pool.
    """
    pool = make_pool(num_blocks=12)
    table = PagedKVPool.block_table_tensor([5, 2, 8], "cpu")
    flat = pool.flat_indices(table, 0, 10)

    mirror = DecodeViewCache(pool.spec, capacity_tokens=16)
    mirror.seed(pool, flat, 10)
    assert mirror.filled == 10

    for layer in range(pool.spec.num_layers):
        ref_k, ref_v = pool.gather_flat(layer, flat)
        got_k, got_v = mirror.view(layer, 10)
        assert torch.equal(ref_k, got_k), f"seeded keys differ, layer {layer}"
        assert torch.equal(ref_v, got_v), f"seeded values differ, layer {layer}"


def test_seed_handles_a_scattered_block_table():
    """Paging's premise: a sequence's blocks are neither contiguous nor ordered."""
    pool = make_pool(num_blocks=20)
    table = PagedKVPool.block_table_tensor([19, 0, 13, 4], "cpu")
    flat = pool.flat_indices(table, 0, 14)

    mirror = DecodeViewCache(pool.spec, capacity_tokens=14)
    mirror.seed(pool, flat, 14)
    ref_k, _ = pool.gather_flat(0, flat)
    got_k, _ = mirror.view(0, 14)
    assert torch.equal(ref_k, got_k)


# ------------------------------------------------------------------- bounds

def test_rejects_overflow():
    pool = make_pool()
    mirror = DecodeViewCache(pool.spec, capacity_tokens=4)
    k = torch.randn(1, pool.spec.num_kv_heads, 3, pool.spec.head_dim)

    with pytest.raises(ValueError, match="exceeds capacity"):
        mirror.append(0, k, k, 2)
    with pytest.raises(ValueError, match="requested 9 tokens"):
        mirror.view(0, 9)

    table = PagedKVPool.block_table_tensor([0, 1, 2], "cpu")
    with pytest.raises(ValueError, match="cannot seed"):
        mirror.seed(pool, pool.flat_indices(table, 0, 9), 9)


def test_truncate_and_reset():
    pool = make_pool()
    mirror = DecodeViewCache(pool.spec, capacity_tokens=8)
    k = torch.randn(1, pool.spec.num_kv_heads, 6, pool.spec.head_dim)
    for layer in range(pool.spec.num_layers):
        mirror.append(layer, k, k, 0)
    assert mirror.filled == 6

    mirror.truncate(4)          # speculative rollback
    assert mirror.filled == 4
    mirror.truncate(99)         # never grows
    assert mirror.filled == 4

    mirror.reset()
    assert mirror.filled == 0
    mirror.invalidate()
    assert mirror.filled == 0


# ------------------------------------------- integration with PagedCache

def test_enabling_the_mirror_does_not_change_a_single_logit(tiny_model):
    """
    The claim that matters.

    Same prompt and decode loop through a real model twice -- once reading the
    pool, once the mirror -- compared exactly.
    """
    spec = ModelKVSpec.from_config(tiny_model.config, torch.float32, "cpu")
    ctx, steps = 23, 7
    num_blocks = (ctx + steps) // BLOCK + 3
    prompt = [2 + (i * 11) % 200 for i in range(ctx)]

    def run(capacity):
        pool = PagedKVPool(spec, num_blocks=num_blocks, block_size=BLOCK)
        cache = PagedCache(pool, list(range(num_blocks)), view_capacity=capacity)
        out = tiny_model(
            input_ids=torch.tensor([prompt], dtype=torch.long),
            past_key_values=cache, use_cache=True,
        )
        logits = []
        for _ in range(steps):
            token = int(out.logits[0, -1].argmax())
            logits.append(out.logits[0, -1].clone())
            cache.begin_step()
            out = tiny_model(
                input_ids=torch.tensor([[token]], dtype=torch.long),
                past_key_values=cache, use_cache=True,
            )
        return logits, cache

    with torch.no_grad():
        plain, plain_cache = run(None)
        mirrored, mirror_cache = run(capacity_for(ctx, steps))

    assert plain_cache.view_cache is None
    assert mirror_cache.view_cache is not None
    for step, (a, b) in enumerate(zip(plain, mirrored)):
        assert torch.equal(a, b), f"logits diverged at decode step {step}"


def test_mirror_is_reseeded_after_a_block_table_change(tiny_model):
    """
    Copy-on-write repoints a logical block at a different physical one, so the
    mirror's contents can no longer be trusted. A rebind must re-read the pool
    rather than keep serving what it already had.
    """
    spec = ModelKVSpec.from_config(tiny_model.config, torch.float32, "cpu")
    ctx = 10
    pool = PagedKVPool(spec, num_blocks=16, block_size=BLOCK)
    cache = PagedCache(pool, [0, 1, 2, 3], view_capacity=32)

    with torch.no_grad():
        tiny_model(
            input_ids=torch.tensor([[2 + i for i in range(ctx)]], dtype=torch.long),
            past_key_values=cache, use_cache=True,
        )
    assert cache.view_cache.filled == ctx

    # Stand in for copy-on-write: copy block 0 to block 9, then repoint.
    pool.key_pool[:, 9] = pool.key_pool[:, 0]
    pool.value_pool[:, 9] = pool.value_pool[:, 0]
    cache.set_block_table([9, 1, 2, 3])

    assert cache.view_cache.filled == ctx, "mirror was not re-seeded after the rebind"
    flat = cache.flat_indices(0, ctx)
    for layer in range(spec.num_layers):
        ref_k, ref_v = pool.gather_flat(layer, flat)
        got_k, got_v = cache.view_cache.view(layer, ctx)
        assert torch.equal(ref_k, got_k), f"stale keys after rebind, layer {layer}"
        assert torch.equal(ref_v, got_v), f"stale values after rebind, layer {layer}"


def test_rollback_keeps_the_mirror_consistent_with_the_cache(tiny_model):
    """Speculative rejection moves the write cursor back; the mirror must follow."""
    spec = ModelKVSpec.from_config(tiny_model.config, torch.float32, "cpu")
    ctx = 12
    pool = PagedKVPool(spec, num_blocks=16, block_size=BLOCK)
    cache = PagedCache(pool, list(range(16)), view_capacity=40)

    with torch.no_grad():
        tiny_model(
            input_ids=torch.tensor([[2 + i for i in range(ctx)]], dtype=torch.long),
            past_key_values=cache, use_cache=True,
        )
    assert cache.view_cache.filled == ctx

    cache.rollback(5)
    assert cache.position == ctx - 5
    assert cache.view_cache.filled == cache.position

    cache.reset()
    assert cache.view_cache.filled == 0


def test_seed_position_fills_the_mirror_for_a_prefix_hit(tiny_model):
    """
    A prefix hit sets the cursor forward without this sequence ever writing
    those positions, so `seed_position` must populate the mirror from the pool.
    """
    spec = ModelKVSpec.from_config(tiny_model.config, torch.float32, "cpu")
    pool = PagedKVPool(spec, num_blocks=16, block_size=BLOCK)
    pool.key_pool.normal_()
    pool.value_pool.normal_()

    cache = PagedCache(pool, list(range(16)), view_capacity=40)
    cache.seed_position(8)

    assert cache.position == 8
    assert cache.view_cache.filled == 8
    flat = cache.flat_indices(0, 8)
    for layer in range(spec.num_layers):
        ref_k, ref_v = pool.gather_flat(layer, flat)
        got_k, got_v = cache.view_cache.view(layer, 8)
        assert torch.equal(ref_k, got_k)
        assert torch.equal(ref_v, got_v)
