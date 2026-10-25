"""
Block-wise paged attention with an online softmax (Phase A, tier 3).

Tiers 1 and 2 make materialising a sequence's KV cheaper. This stops
materialising it. Attention is computed one 16-token block at a time directly
out of the pool, carrying a running maximum and sum — the FlashAttention
formulation — so working memory is `O(block_size)` rather than `O(context)` and
no intermediate context tensor is ever allocated.

That is what "fused kernel" means in
[PROGRESS_AND_NEXT_PHASES.md](../../PROGRESS_AND_NEXT_PHASES.md) §8 Phase A, and
it is the only tier that does not trade memory for speed: tier 2's mirror costs
a contiguous copy per sequence, and this costs nothing.

**This is a reference implementation, and it is slower than tiers 1-2 on CPU.**
That is expected and is not a reason to discard it. Python-level iteration over
blocks pays interpreter overhead per block, which the vectorised whole-context
gather avoids; the structure only pays off when the per-block work runs on a
device that can hide that overhead. Its value here is threefold:

  1. **It is the shape a Triton/CUDA kernel takes.** Phase E ports this loop
     body, not the gather.
  2. **The equivalence proof is the artefact.** A correct, slow kernel that is
     provably identical to standard attention is worth more than a fast one
     whose output nobody checked, because a wrong KV read produces fluent,
     confident, wrong text rather than a crash.
  3. **It bounds working memory**, which is the property that makes very long
     contexts feasible at all.

The online softmax identity being used, for one query against blocks
`B_1..B_n`: keeping a running max `m` and sum `l`, after absorbing a block with
local max `m'` and local sum `l'`,

    m_new = max(m, m')
    l_new = l * exp(m - m_new) + l' * exp(m' - m_new)
    acc_new = acc * exp(m - m_new) + acc' * exp(m' - m_new)

and the final output is `acc / l`. Subtracting the running max before
exponentiating is what keeps it numerically stable — the whole reason not to
simply sum `exp(scores)`.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def paged_attention_blockwise(
    query: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    layer_idx: int,
    block_table: torch.Tensor,
    context_length: int,
    block_size: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Attention for a single decode step, reading the pool block by block.

    Args:
        query: `(1, num_heads, 1, head_dim)` — one query position, the decode
            case. `num_heads` may exceed the pool's `num_kv_heads`, in which
            case grouped-query attention is applied by repeating KV heads.
        key_pool, value_pool: `(layers, blocks, slots, kv_heads, head_dim)`.
        block_table: physical block id per logical block, `(num_logical,)`.
        context_length: valid tokens; the final block is usually partial.
        scale: defaults to `1/sqrt(head_dim)`.

    Returns:
        `(1, num_heads, 1, head_dim)`.

    Working memory is one block's worth of scores, independent of
    `context_length`.
    """
    if query.shape[0] != 1 or query.shape[2] != 1:
        raise ValueError(
            f"expected a single query position (1, H, 1, D), got {tuple(query.shape)}"
        )
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")

    num_heads, head_dim = query.shape[1], query.shape[3]
    num_kv_heads = key_pool.shape[3]
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"{num_heads} query heads is not a multiple of {num_kv_heads} kv heads"
        )
    group = num_heads // num_kv_heads
    scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)

    q = query[0, :, 0, :]  # (H, D)
    dtype = torch.float32  # accumulate in fp32 regardless of pool dtype

    # Running online-softmax state, per head.
    running_max = torch.full((num_heads,), float("-inf"), dtype=dtype, device=q.device)
    running_sum = torch.zeros((num_heads,), dtype=dtype, device=q.device)
    accumulator = torch.zeros((num_heads, head_dim), dtype=dtype, device=q.device)

    num_logical = (context_length + block_size - 1) // block_size
    if num_logical > block_table.numel():
        raise IndexError(
            f"context of {context_length} tokens needs {num_logical} blocks, "
            f"block table has {block_table.numel()}"
        )

    for logical in range(num_logical):
        physical = int(block_table[logical])
        # The last block is usually partial; reading its unwritten slots would
        # mix uninitialised KV into the softmax.
        valid = min(block_size, context_length - logical * block_size)

        k_block = key_pool[layer_idx, physical, :valid].to(dtype)    # (valid, Hkv, D)
        v_block = value_pool[layer_idx, physical, :valid].to(dtype)

        if group > 1:
            # Grouped-query attention: each KV head serves `group` query heads.
            k_block = k_block.repeat_interleave(group, dim=1)
            v_block = v_block.repeat_interleave(group, dim=1)

        # (H, valid): this query against every key in the block.
        scores = torch.einsum("hd,thd->ht", q.to(dtype), k_block) * scale

        block_max = scores.max(dim=-1).values                        # (H,)
        new_max = torch.maximum(running_max, block_max)
        # exp(-inf - -inf) is nan, so the first block is handled by the fact
        # that running_sum is 0 and its rescale factor is multiplied by 0.
        rescale_old = torch.where(
            torch.isinf(running_max),
            torch.zeros_like(running_max),
            torch.exp(running_max - new_max),
        )
        probs = torch.exp(scores - new_max.unsqueeze(-1))             # (H, valid)

        running_sum = running_sum * rescale_old + probs.sum(dim=-1)
        accumulator = accumulator * rescale_old.unsqueeze(-1) + torch.einsum(
            "ht,thd->hd", probs, v_block
        )
        running_max = new_max

    out = accumulator / running_sum.unsqueeze(-1)
    return out.unsqueeze(0).unsqueeze(2).to(query.dtype)


def reference_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Standard single-query attention over a materialised context.

    Exists to be the thing `paged_attention_blockwise` is proved equal to. Takes
    `(1, H, 1, D)` query and `(1, Hkv, T, D)` keys/values, applying the same
    grouped-query expansion.
    """
    num_heads, head_dim = query.shape[1], query.shape[3]
    num_kv_heads = keys.shape[1]
    group = num_heads // num_kv_heads
    scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)

    k = keys.to(torch.float32)
    v = values.to(torch.float32)
    if group > 1:
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)

    scores = torch.einsum("bhqd,bhtd->bhqt", query.to(torch.float32), k) * scale
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqt,bhtd->bhqd", weights, v).to(query.dtype)
