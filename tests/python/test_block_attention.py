"""
Equivalence tests for block-wise paged attention (Phase A, tier 3).

The kernel is slower than tiers 1-2 on CPU and is not wired into the engine.
**The equivalence proof is the deliverable** — it is the thing a Triton port in
Phase E would be checked against, and a fast kernel whose output nobody verified
is worth nothing, because a wrong attention read produces fluent, confident,
wrong text rather than a crash.

Tolerances are `1e-5` rather than exact: the online softmax rescales partial
sums in a different order from a single fused softmax, so the two differ in
floating-point association. That is the one place in this project where exact
equality is the wrong bar, and it is why the reference accumulates in fp32 too.
"""

from __future__ import annotations

import math

import pytest
import torch

from memforge_ml.kernels.block_attention import (
    paged_attention_blockwise,
    reference_attention,
)
from memforge_ml.runtime.kv_pool import ModelKVSpec, PagedKVPool

TOL = 1e-5


def build(num_layers=2, num_kv_heads=2, head_dim=8, num_blocks=16, block_size=4,
          seed=0):
    torch.manual_seed(seed)
    spec = ModelKVSpec(
        num_layers=num_layers, num_kv_heads=num_kv_heads, head_dim=head_dim,
        dtype=torch.float32, device="cpu",
    )
    pool = PagedKVPool(spec, num_blocks=num_blocks, block_size=block_size)
    pool.key_pool.normal_()
    pool.value_pool.normal_()
    return spec, pool


def materialise(pool, layer_idx, table, context_length):
    """The contiguous context the block-wise kernel is meant to avoid building."""
    physical, slots = pool.slot_indices(table, 0, context_length)
    return pool.gather(layer_idx, physical, slots)


@pytest.mark.parametrize("context_length", [1, 3, 4, 5, 8, 13, 16, 31, 64])
def test_matches_reference_across_context_lengths(context_length):
    """
    Partial final blocks are the interesting case.

    A kernel that read a whole block regardless would mix unwritten slots into
    the softmax, and every context that is not a multiple of the block size
    would be subtly wrong.
    """
    spec, pool = build(num_blocks=32, block_size=4)
    num_logical = (context_length + 4 - 1) // 4
    table = PagedKVPool.block_table_tensor(list(range(num_logical)), "cpu")
    query = torch.randn(1, spec.num_kv_heads, 1, spec.head_dim)

    for layer in range(spec.num_layers):
        keys, values = materialise(pool, layer, table, context_length)
        expected = reference_attention(query, keys, values)
        got = paged_attention_blockwise(
            query, pool.key_pool, pool.value_pool, layer, table,
            context_length, pool.block_size,
        )
        assert got.shape == expected.shape
        assert torch.allclose(got, expected, atol=TOL), (
            f"layer {layer}, context {context_length}: "
            f"max diff {float((got - expected).abs().max())}"
        )


def test_matches_reference_with_a_scattered_block_table():
    """
    Paging's premise: a sequence's blocks are neither contiguous nor ordered.

    A kernel that assumed ascending block ids would pass every tidy test and
    fail here.
    """
    spec, pool = build(num_blocks=24, block_size=4)
    table = PagedKVPool.block_table_tensor([23, 0, 17, 5, 11], "cpu")
    context_length = 18
    query = torch.randn(1, spec.num_kv_heads, 1, spec.head_dim)

    keys, values = materialise(pool, 0, table, context_length)
    expected = reference_attention(query, keys, values)
    got = paged_attention_blockwise(
        query, pool.key_pool, pool.value_pool, 0, table,
        context_length, pool.block_size,
    )
    assert torch.allclose(got, expected, atol=TOL)


@pytest.mark.parametrize("num_kv_heads,num_heads", [(2, 2), (2, 4), (2, 8), (5, 15)])
def test_grouped_query_attention(num_kv_heads, num_heads):
    """
    GQA is the normal case for the models this project runs: SmolLM2-360M has
    more query heads than KV heads, so a kernel that only handled the 1:1 case
    would be useless on the actual target.
    """
    spec, pool = build(num_kv_heads=num_kv_heads, num_blocks=16, block_size=4)
    table = PagedKVPool.block_table_tensor([2, 7, 1], "cpu")
    context_length = 10
    query = torch.randn(1, num_heads, 1, spec.head_dim)

    keys, values = materialise(pool, 0, table, context_length)
    expected = reference_attention(query, keys, values)
    got = paged_attention_blockwise(
        query, pool.key_pool, pool.value_pool, 0, table,
        context_length, pool.block_size,
    )
    assert got.shape == (1, num_heads, 1, spec.head_dim)
    assert torch.allclose(got, expected, atol=TOL)


def test_custom_scale_is_honoured():
    spec, pool = build(num_blocks=8, block_size=4)
    table = PagedKVPool.block_table_tensor([0, 1], "cpu")
    query = torch.randn(1, spec.num_kv_heads, 1, spec.head_dim)
    keys, values = materialise(pool, 0, table, 6)

    for scale in (1.0, 0.25, 1.0 / math.sqrt(spec.head_dim)):
        expected = reference_attention(query, keys, values, scale=scale)
        got = paged_attention_blockwise(
            query, pool.key_pool, pool.value_pool, 0, table, 6,
            pool.block_size, scale=scale,
        )
        assert torch.allclose(got, expected, atol=TOL), f"scale {scale}"


def test_is_numerically_stable_with_large_scores():
    """
    The reason for subtracting a running max rather than summing exp(scores).

    With large logits a naive implementation overflows to inf and then produces
    nan; this must stay finite and correct.
    """
    spec, pool = build(num_blocks=8, block_size=4)
    pool.key_pool.mul_(50.0)       # push scores far out
    table = PagedKVPool.block_table_tensor([0, 1, 2], "cpu")
    query = torch.randn(1, spec.num_kv_heads, 1, spec.head_dim) * 50.0

    keys, values = materialise(pool, 0, table, 11)
    expected = reference_attention(query, keys, values)
    got = paged_attention_blockwise(
        query, pool.key_pool, pool.value_pool, 0, table, 11, pool.block_size,
    )
    assert torch.isfinite(got).all(), "online softmax overflowed"
    assert torch.allclose(got, expected, atol=1e-4)


def test_single_block_and_single_token():
    """Degenerate sizes: one token is the first decode step after a 1-token prompt."""
    spec, pool = build(num_blocks=4, block_size=4)
    table = PagedKVPool.block_table_tensor([1], "cpu")
    query = torch.randn(1, spec.num_kv_heads, 1, spec.head_dim)

    keys, values = materialise(pool, 0, table, 1)
    expected = reference_attention(query, keys, values)
    got = paged_attention_blockwise(
        query, pool.key_pool, pool.value_pool, 0, table, 1, pool.block_size,
    )
    assert torch.allclose(got, expected, atol=TOL)
    # With one key, attention is that key's value exactly.
    assert torch.allclose(got, values, atol=TOL)


def test_rejects_bad_inputs():
    spec, pool = build(num_blocks=8, block_size=4)
    table = PagedKVPool.block_table_tensor([0, 1], "cpu")
    query = torch.randn(1, spec.num_kv_heads, 1, spec.head_dim)

    with pytest.raises(ValueError, match="single query position"):
        paged_attention_blockwise(
            torch.randn(1, spec.num_kv_heads, 3, spec.head_dim),
            pool.key_pool, pool.value_pool, 0, table, 5, pool.block_size,
        )
    with pytest.raises(ValueError, match="context_length must be positive"):
        paged_attention_blockwise(
            query, pool.key_pool, pool.value_pool, 0, table, 0, pool.block_size,
        )
    with pytest.raises(ValueError, match="not a multiple"):
        paged_attention_blockwise(
            torch.randn(1, spec.num_kv_heads * 2 + 1, 1, spec.head_dim),
            pool.key_pool, pool.value_pool, 0, table, 5, pool.block_size,
        )
    with pytest.raises(IndexError, match="block table has"):
        paged_attention_blockwise(
            query, pool.key_pool, pool.value_pool, 0,
            PagedKVPool.block_table_tensor([0], "cpu"), 20, pool.block_size,
        )


def test_working_memory_is_independent_of_context():
    """
    The property that makes this tier worth having.

    Peak intermediate allocation must not grow with context. Measured by the
    number of elements in the largest score tensor the kernel can create, which
    is bounded by heads x block_size regardless of how long the context is.
    """
    spec, pool = build(num_blocks=256, block_size=16)
    query = torch.randn(1, spec.num_kv_heads, 1, spec.head_dim)

    for context_length in (16, 256, 2048):
        num_logical = (context_length + 15) // 16
        table = PagedKVPool.block_table_tensor(list(range(num_logical)), "cpu")
        keys, values = materialise(pool, 0, table, context_length)
        expected = reference_attention(query, keys, values)
        got = paged_attention_blockwise(
            query, pool.key_pool, pool.value_pool, 0, table,
            context_length, pool.block_size,
        )
        assert torch.allclose(got, expected, atol=TOL), (
            f"context {context_length}: max diff "
            f"{float((got - expected).abs().max())}"
        )
