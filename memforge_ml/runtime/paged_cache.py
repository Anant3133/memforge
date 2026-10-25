"""
PagedCache: a `transformers.Cache` implementation backed by the paged KV pool.

This is the component that makes PagedAttention real rather than simulated --
dropping a `PagedCache` into `model(..., past_key_values=cache)` means every
attention layer of an unmodified Hugging Face model writes its Keys/Values into
16-token physical blocks handed out by the C++ `BlockAllocator`, and reads them
back through the logical -> physical block table.

Two properties come for free from paging and are relied on elsewhere:
  * growth is block-granular -- a sequence never needs a contiguous reservation;
  * `rollback(n)` is O(1) -- rejecting speculative tokens just moves the write
    cursor back, with no tensor copies or reallocation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from memforge_ml.runtime.decode_view_cache import DecodeViewCache
from memforge_ml.runtime.kv_pool import PagedKVPool


class PagedCacheLayer(CacheLayerMixin):
    """One attention layer's view of the shared paged pool."""

    is_sliding = False
    is_compileable = False
    supports_early_init = False

    def __init__(self, owner: "PagedCache", layer_idx: int):
        super().__init__()
        self.owner = owner
        self.layer_idx = layer_idx
        self.position = 0
        self.is_initialized = True

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        # The pool is allocated up front, so there is nothing to initialize lazily.
        self.is_initialized = True

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if key_states.shape[0] != 1:
            raise ValueError(
                f"PagedCache is a single-sequence cache; got batch size {key_states.shape[0]}. "
                "Batched decoding goes through ContinuousBatchingEngine instead."
            )

        num_new = key_states.shape[-2]
        start, end = self.position, self.position + num_new

        # Flat indices: one index_select / index_copy_ per tensor instead of a
        # two-dimensional advanced index. Memoised across layers, so the
        # translation is paid once per step rather than 32 times.
        write_index = self.owner.flat_indices(start, end)
        self.owner.pool.write_flat(self.layer_idx, write_index, key_states, value_states)

        if self.owner.lazy_read and num_new == 1:
            # Lazy mode: hand back only the token just written. The attention
            # backend reads the rest of the context out of the pool block by
            # block, which is what keeps working memory O(block_size) instead of
            # O(context) -- materialising the full context here would make the
            # block-wise kernel pointless.
            #
            # Restricted to num_new == 1 because that is the case the backend
            # handles: a single query attends to everything before it and needs
            # no mask. Prefill still returns the full context below.
            self.position = end
            return key_states, value_states

        mirror = self.owner.view_cache
        if mirror is not None:
            # The pool above is still the source of truth; this mirrors the same
            # write so the read below is a view instead of a fresh gather of the
            # whole context. O(1) per step rather than O(context).
            mirror.append(self.layer_idx, key_states, value_states, start)
            self.position = end
            return mirror.view(self.layer_idx, end)

        self.position = end
        read_index = self.owner.flat_indices(0, end)
        return self.owner.pool.gather_flat(self.layer_idx, read_index)

    def get_mask_sizes(self, query_length: int) -> Tuple[int, int]:
        return self.position + query_length, 0

    def get_seq_length(self) -> int:
        return self.position

    def get_max_length(self) -> int:
        return self.owner.pool.capacity_tokens

    def reset(self) -> None:
        self.position = 0

    def crop(self, tokens_to_remove: int) -> None:
        self.position = max(0, self.position - abs(tokens_to_remove))


class PagedCache(Cache):
    """
    Single-sequence KV cache whose storage is the shared `PagedKVPool`.

    The owning engine must call `set_block_table()` whenever the C++ allocator
    grows the sequence's block table, and `begin_step()` before each forward pass
    so the per-step index translation is computed once and shared by all layers.
    """

    def __init__(
        self,
        pool: PagedKVPool,
        block_table: List[int],
        device: Optional[str] = None,
        view_capacity: Optional[int] = None,
        lazy_read: bool = False,
    ):
        """
        `view_capacity`, when given, enables the append-only decode mirror
        (`DecodeViewCache`) sized to that many tokens. Off by default: it is a
        contiguous copy of this sequence's KV, so it trades the memory paging
        exists to save for a measured ~10x on the decode read path. See
        decode_view_cache.py for the arithmetic.

        `lazy_read` makes a single-token `update()` return only that token
        rather than the whole context, for use with the block-wise attention
        backend which reads the context from the pool itself. It is mutually
        exclusive with `view_capacity`: the mirror exists to make materialising
        cheap, and lazy mode exists to avoid materialising at all.
        """
        if lazy_read and view_capacity:
            raise ValueError(
                "lazy_read and view_capacity are mutually exclusive: the mirror "
                "materialises the context, lazy mode avoids materialising it"
            )
        self.pool = pool
        self.device_str = device or pool.spec.device
        self._block_table: torch.Tensor = PagedKVPool.block_table_tensor(block_table, self.device_str)
        self._index_memo: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._flat_memo: Dict[Tuple[int, int], torch.Tensor] = {}
        self.lazy_read = bool(lazy_read)
        self.view_cache: Optional[DecodeViewCache] = (
            DecodeViewCache(pool.spec, view_capacity, self.device_str)
            if view_capacity
            else None
        )
        super().__init__(layers=[PagedCacheLayer(self, i) for i in range(pool.spec.num_layers)])

    # ------------------------------------------------------- engine interface

    def set_block_table(self, block_table: List[int]) -> None:
        """Re-bind to a (possibly grown) logical -> physical block table."""
        if self._block_table.numel() != len(block_table) or not torch.equal(
            self._block_table, PagedKVPool.block_table_tensor(block_table, self.device_str)
        ):
            self._block_table = PagedKVPool.block_table_tensor(block_table, self.device_str)
            self._index_memo.clear()
            self._flat_memo.clear()
            if self.view_cache is not None:
                # The table changed, which after copy-on-write means some logical
                # position now lives in a different physical block. The mirror
                # cannot tell which, so it re-reads what it needs from the pool
                # -- the only safe move, and rare enough not to matter.
                resident = self.layers[0].position
                self.view_cache.invalidate()
                if resident:
                    self.view_cache.seed(
                        self.pool, self.flat_indices(0, resident), resident
                    )

    def begin_step(self, block_table: Optional[List[int]] = None) -> None:
        if block_table is not None:
            self.set_block_table(block_table)
        self._index_memo.clear()
        self._flat_memo.clear()

    def seed_position(self, num_cached_tokens: int) -> None:
        """
        Declare that the first `num_cached_tokens` positions are already resident.

        Used by partial prefill after a prefix-cache hit: those tokens' Keys and
        Values are already in the shared physical blocks, so the forward pass
        feeds only the suffix. Moving the write cursor forward is what keeps the
        suffix from landing on top of the cached prefix -- which would corrupt it
        for every other sequence pointing at those same blocks.

        `get_mask_sizes()` derives the causal mask span from `position`, so
        seeding it also makes the mask cover cached + new without further work.
        """
        if num_cached_tokens < 0:
            raise ValueError(f"num_cached_tokens must be non-negative, got {num_cached_tokens}")
        for layer in self.layers:
            layer.position = num_cached_tokens
        if self.view_cache is not None:
            # These tokens were written by a *different* sequence, whose blocks
            # this one now shares, so the mirror has never seen them. Fill it
            # from the pool once rather than leaving it to serve zeros.
            self.view_cache.invalidate()
            if num_cached_tokens:
                self.view_cache.seed(
                    self.pool, self.flat_indices(0, num_cached_tokens), num_cached_tokens
                )
        self._index_memo.clear()
        self._flat_memo.clear()

    def indices(self, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Index translation for token range [start, end), memoized across layers."""
        key = (start, end)
        if key not in self._index_memo:
            self._index_memo[key] = self.pool.slot_indices(self._block_table, start, end)
        return self._index_memo[key]

    def flat_indices(self, start: int, end: int) -> torch.Tensor:
        """
        Flattened pool offsets for [start, end), memoized across layers.

        Separate memo from `indices` because the two forms are used by different
        callers: the batched decode path still needs (physical, slots) pairs,
        while the per-layer read wants the collapsed offset.
        """
        key = (start, end)
        if key not in self._flat_memo:
            self._flat_memo[key] = self.pool.flat_indices(self._block_table, start, end)
        return self._flat_memo[key]

    @property
    def block_table(self) -> List[int]:
        return self._block_table.tolist()

    def block_table_tensor_cached(self) -> torch.Tensor:
        """
        The block table as the tensor it is already stored as.

        `block_table` returns a Python list, which the batched decode path would
        immediately have to turn back into a tensor once per sequence per step.
        This hands out the existing tensor instead. Treat it as read-only --
        mutating it would desynchronise the cache from the C++ block table.
        """
        return self._block_table

    @property
    def position(self) -> int:
        return self.layers[0].position

    def rollback(self, num_tokens: int) -> None:
        """
        Discard the last `num_tokens` cached positions. O(1): only the write
        cursor moves -- the physical blocks stay allocated and are overwritten
        by the next token. This is what makes speculative rejection cheap.
        """
        if num_tokens <= 0:
            return
        for layer in self.layers:
            layer.crop(num_tokens)
        if self.view_cache is not None:
            self.view_cache.truncate(self.layers[0].position)
        self._index_memo.clear()
        self._flat_memo.clear()

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()
        self._index_memo.clear()
        self._flat_memo.clear()
        if self.view_cache is not None:
            self.view_cache.reset()

    def num_blocks_needed(self, total_tokens: int) -> int:
        bs = self.pool.block_size
        return (total_tokens + bs - 1) // bs
