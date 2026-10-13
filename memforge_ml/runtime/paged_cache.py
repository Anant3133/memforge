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

        write_phys, write_slots = self.owner.indices(start, end)
        self.owner.pool.write(self.layer_idx, write_phys, write_slots, key_states, value_states)
        self.position = end

        read_phys, read_slots = self.owner.indices(0, end)
        return self.owner.pool.gather(self.layer_idx, read_phys, read_slots)

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

    def __init__(self, pool: PagedKVPool, block_table: List[int], device: Optional[str] = None):
        self.pool = pool
        self.device_str = device or pool.spec.device
        self._block_table: torch.Tensor = PagedKVPool.block_table_tensor(block_table, self.device_str)
        self._index_memo: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        super().__init__(layers=[PagedCacheLayer(self, i) for i in range(pool.spec.num_layers)])

    # ------------------------------------------------------- engine interface

    def set_block_table(self, block_table: List[int]) -> None:
        """Re-bind to a (possibly grown) logical -> physical block table."""
        if self._block_table.numel() != len(block_table) or not torch.equal(
            self._block_table, PagedKVPool.block_table_tensor(block_table, self.device_str)
        ):
            self._block_table = PagedKVPool.block_table_tensor(block_table, self.device_str)
            self._index_memo.clear()

    def begin_step(self, block_table: Optional[List[int]] = None) -> None:
        if block_table is not None:
            self.set_block_table(block_table)
        self._index_memo.clear()

    def indices(self, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Index translation for token range [start, end), memoized across layers."""
        key = (start, end)
        if key not in self._index_memo:
            self._index_memo[key] = self.pool.slot_indices(self._block_table, start, end)
        return self._index_memo[key]

    @property
    def block_table(self) -> List[int]:
        return self._block_table.tolist()

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
        self._index_memo.clear()

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()
        self._index_memo.clear()

    def num_blocks_needed(self, total_tokens: int) -> int:
        bs = self.pool.block_size
        return (total_tokens + bs - 1) // bs
