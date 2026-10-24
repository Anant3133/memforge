"""
Python half of prefix sharing: pool-level copy-on-write and sharing telemetry.

The decision about *which* blocks are shared lives in C++ (`memforge_c.PrefixCache`),
because that is where the allocator and its reference counts live. What stays
here is everything that has to touch the KV pool tensors or report to the UI:

  * `copy_block()` -- the data movement half of copy-on-write. The C++ side
    repoints a sequence's block table at a private replacement; this copies the
    old block's Keys and Values into it, so the write that follows does not
    destroy a prefix another sequence is still reading.
  * `sharing_telemetry()` / `block_owners()` -- the numbers the dashboard needs
    to show prefix reuse, including the multi-owner block map that a flat
    "one owner per block" occupancy list cannot express.

These are kept out of `kv_pool.py` deliberately: that file is on the critical
path of the paged-gather kernel work, and keeping prefix sharing out of it means
the two tracks do not edit the same file.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from memforge_ml.runtime.kv_pool import PagedKVPool


def copy_block(pool: PagedKVPool, src: int, dst: int) -> None:
    """
    Copy one physical block's KV contents, across every layer, from `src` to `dst`.

    This is the data half of copy-on-write. It must run *before* the write that
    triggered the CoW, or the private copy starts life with a hole in it.

    The pool is laid out (layers, blocks, slots, kv_heads, head_dim), so indexing
    `[:, block]` selects that block in all layers at once and the copy is two
    slice assignments rather than a per-layer Python loop.
    """
    if src == dst:
        return
    pool.key_pool[:, dst] = pool.key_pool[:, src]
    pool.value_pool[:, dst] = pool.value_pool[:, src]


def block_owners(
    runtimes: Iterable[Tuple[int, List[int]]],
    total_blocks: int,
) -> Dict[str, List[int]]:
    """
    Map each allocated physical block to *every* sequence referencing it.

    `runtimes` yields (seq_id, block_table) pairs. Shared blocks appear with more
    than one owner, which is the whole point -- the existing flat occupancy list
    reports a single owner per block and so cannot show sharing at all. Keys are
    strings because this goes out as JSON.
    """
    owners: Dict[str, List[int]] = {}
    for seq_id, table in runtimes:
        for block_id in table:
            if 0 <= block_id < total_blocks:
                owners.setdefault(str(block_id), []).append(seq_id)
    for ids in owners.values():
        ids.sort()
    return owners


def sharing_telemetry(
    allocator: Any,
    pool: PagedKVPool,
    prefix_cache: Optional[Any],
    owners: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, Any]:
    """
    Prefix-sharing metrics, additive to the existing telemetry payload.

    Returns zeros rather than omitting keys when sharing is disabled, so the
    dashboard can bind to a stable shape and show "0 MB saved" instead of
    breaking on a missing field.

    `memory_saved_mb` is the honest figure: blocks that sharing spared the pool
    (sum of reference counts minus distinct allocated blocks), converted at the
    pool's real bytes-per-block. It is what the allocation would have cost
    without sharing, minus what it actually cost.
    """
    blocks_saved = allocator.get_num_blocks_saved()
    payload: Dict[str, Any] = {
        "shared_blocks": allocator.get_num_shared_blocks(),
        "blocks_saved": blocks_saved,
        "memory_saved_mb": round(blocks_saved * pool.bytes_per_block / (1024 * 1024), 3),
        "cached_prefix_blocks": prefix_cache.num_cached_blocks() if prefix_cache else 0,
        "prefix_hit_rate": round(prefix_cache.hit_rate(), 4) if prefix_cache else 0.0,
        "prefix_cache_hits": prefix_cache.num_hits() if prefix_cache else 0,
        "prefix_cache_misses": prefix_cache.num_misses() if prefix_cache else 0,
        "prefix_sharing_enabled": prefix_cache is not None,
    }
    if owners is not None:
        payload["block_owners"] = owners
    return payload
