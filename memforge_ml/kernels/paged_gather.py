"""
Paged KV gather and scatter (Phase A, tier 1).

The pool is laid out `(layers, blocks, slots, kv_heads, head_dim)` and attention
wants `(1, kv_heads, tokens, head_dim)`, so every read costs an index plus a
transpose. This module makes that read as cheap as a layout-respecting
implementation can, and nothing more — the remaining cost is structural and is
addressed by `batched_decode.py` and `decode_view_cache.py`.

What was measured before writing this (SmolLM2-360M geometry, 32 layers, 5 KV
heads, head_dim 64, fp32, block size 16, real 320 MB pool):

| context | current gather | of which permute+contiguous |
| ------: | -------------: | --------------------------: |
|     128 |        22.6 ms |                      6.2 ms |
|     512 |        26.7 ms |                      8.7 ms |
|    1024 |        41.1 ms |                     15.2 ms |
|    2048 |        79.4 ms |                     25.5 ms |

Two things that guided the design, both of which contradict the obvious guess:

  * **A preallocated destination buffer is slower.** Writing through a
    transposed view (`dest.transpose(0,1).copy_(src)`) measured 0.93x at 2048 —
    i.e. slower than `permute().contiguous()`, which dispatches to an optimised
    copy kernel. The Phase A plan suggested exactly this, so it is recorded here
    rather than silently skipped.
  * **Flattening the index helps.** Collapsing `(block, slot)` into a single
    `block * block_size + slot` offset turns a two-dimensional advanced index
    into one `index_select` over a flat view, worth a consistent ~1.2x.

Index arithmetic itself is not worth optimising: building the index tensors
costs 0.049 ms at 2048 tokens, 0.06% of the gather.
"""

from __future__ import annotations

from typing import Tuple

import torch


def flat_slot_index(physical: torch.Tensor, slots: torch.Tensor,
                    block_size: int) -> torch.Tensor:
    """
    Collapse `(physical_block, slot)` into one offset into a flattened pool.

    `pool[layer]` has shape `(blocks, slots, heads, dim)`; viewed as
    `(blocks * slots, heads, dim)` the element for `(block, slot)` sits at
    `block * block_size + slot`. That makes the gather a single `index_select`
    on dimension 0 instead of a two-dimensional advanced index.

    Returned contiguous because `index_select` wants a contiguous index tensor
    and would otherwise copy it on every call.
    """
    return (physical * block_size + slots).contiguous()


def gather_kv(
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    layer_idx: int,
    flat_index: torch.Tensor,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Read a sequence's K/V for one layer as `(1, heads, tokens, head_dim)`.

    Equivalent to `PagedKVPool.gather`, and tested against it for byte-equality.
    `flat_index` comes from `flat_slot_index`.

    The `permute(1, 0, 2).unsqueeze(0).contiguous()` is deliberate and is not an
    oversight: it is the fastest route from the pool's `(tokens, heads, dim)` to
    attention's `(1, heads, tokens, dim)` that was found. Writing into a
    preallocated transposed view instead measured *slower* (see the module
    docstring), so the copy stays.
    """
    num_blocks, slots_per_block = key_pool.shape[1], key_pool.shape[2]
    assert slots_per_block == block_size, (
        f"pool block size {slots_per_block} does not match {block_size}"
    )
    flat_k = key_pool[layer_idx].view(num_blocks * slots_per_block, *key_pool.shape[3:])
    flat_v = value_pool[layer_idx].view(num_blocks * slots_per_block, *value_pool.shape[3:])

    k = torch.index_select(flat_k, 0, flat_index)
    v = torch.index_select(flat_v, 0, flat_index)
    return (
        k.permute(1, 0, 2).unsqueeze(0).contiguous(),
        v.permute(1, 0, 2).unsqueeze(0).contiguous(),
    )


def scatter_kv(
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    layer_idx: int,
    flat_index: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    block_size: int,
) -> None:
    """
    Write `(1, heads, tokens, head_dim)` K/V back into the pool, in place.

    The mirror of `gather_kv`, equivalent to `PagedKVPool.write`. Uses
    `index_copy_` on the flattened view, which is the write counterpart of
    `index_select` and avoids materialising an intermediate index grid.
    """
    num_blocks, slots_per_block = key_pool.shape[1], key_pool.shape[2]
    flat_k = key_pool[layer_idx].view(num_blocks * slots_per_block, *key_pool.shape[3:])
    flat_v = value_pool[layer_idx].view(num_blocks * slots_per_block, *value_pool.shape[3:])

    # (1, H, T, D) -> (T, H, D), matching the pool's slot-major layout.
    k = keys[0].permute(1, 0, 2).to(flat_k.dtype).contiguous()
    v = values[0].permute(1, 0, 2).to(flat_v.dtype).contiguous()

    flat_k.index_copy_(0, flat_index, k)
    flat_v.index_copy_(0, flat_index, v)
