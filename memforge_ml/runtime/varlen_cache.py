"""
A cache for one fused prefill+decode forward pass (Phase C).

`PagedCache` is explicitly a *single-sequence* cache -- it refuses a batch
dimension above 1 -- and that is what forces the engine to run a prefill chunk
and a decode step as two passes through the model. Fusing them needs a cache
that accepts one packed run of query tokens belonging to different sequences and
routes each token's K/V to that sequence's own physical blocks.

**Packed, not batched.** The tokens arrive as a single row of shape
`(1, heads, T_total, head_dim)`, with two parallel metadata arrays saying which
sequence each token belongs to and what its absolute position is. There is no
padding, so no mask hides anything and no compute is wasted on it -- which is
the point, since the alternative for a 256-token chunk beside three decode
tokens is a batch padded to 256 wide and 255/256 of it discarded.

**Always lazy.** `update()` scatters into the pool and returns the new K/V
unchanged; it never materialises a context. It cannot, in fact: the tokens in
one pass belong to different sequences with different contexts, so there is no
single tensor to return. The reader is `paged_attention_varlen`, which goes to
the pool block by block and does its own per-token causal masking from the
positions. That makes the pool the only source of truth here, with no mirror and
no gathered context.

**Why the attention backend must not fall back for this cache.** Transformers
builds a causal mask sized for one sequence of `T_total` tokens, and for a
packed run that mask is simply wrong -- it would let one sequence's tokens
attend to another's. The varlen kernel ignores it and masks from positions
instead, which is correct, but it means a silent fallback to eager attention
would compute a confidently wrong answer rather than a slow one. So the varlen
binding raises instead of falling back.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
from transformers.cache_utils import Cache, CacheLayerMixin


class VarlenPackedCacheLayer(CacheLayerMixin):
    """One layer's view of a packed, multi-sequence write."""

    is_sliding = False
    is_compileable = False
    supports_early_init = False

    def __init__(self, owner: "VarlenPackedCache", layer_idx: int):
        super().__init__()
        self.owner = owner
        self.layer_idx = layer_idx
        self.is_initialized = True

    def lazy_initialization(
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> None:
        # The pool is allocated up front; nothing to initialise lazily.
        self.is_initialized = True

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scatter every packed token's K/V into its own sequence's slot.

        Writing happens here, *before* the attention interface runs, which is
        what lets a token attend to itself: by the time the kernel reads the
        pool, this step's keys are in it. The same ordering is what
        `attention_backend`'s single-sequence binding relies on.
        """
        if key_states.shape[0] != 1:
            raise ValueError(
                f"VarlenPackedCache takes one packed row, got batch "
                f"{key_states.shape[0]}"
            )
        if key_states.shape[-2] != self.owner.num_tokens:
            raise ValueError(
                f"{key_states.shape[-2]} key states for "
                f"{self.owner.num_tokens} packed tokens"
            )

        self.owner.pool.write_flat(
            self.layer_idx, self.owner.write_index, key_states, value_states
        )
        # Lazy by construction: there is no single context to return, because
        # these tokens belong to different sequences.
        return key_states, value_states

    def get_mask_sizes(self, query_length: int) -> Tuple[int, int]:
        # Keeps any mask transformers insists on building down to the packed run
        # itself rather than the longest context in it. The varlen kernel does
        # not read that mask -- see this module's docstring.
        return query_length, 0

    def get_seq_length(self) -> int:
        # No single sequence length exists for a packed run. Reporting 0 keeps
        # transformers from offsetting positions on its own; the engine supplies
        # absolute `position_ids` explicitly.
        return 0

    def get_max_length(self) -> int:
        return self.owner.pool.capacity_tokens

    def reset(self) -> None:
        pass

    def crop(self, tokens_to_remove: int) -> None:
        # A packed pass is written straight through to the pool and is never
        # rolled back: the engine advances each sequence's own cursor after the
        # pass, and undoing a fused step means not advancing them.
        raise NotImplementedError(
            "VarlenPackedCache holds no per-sequence cursor to crop; roll back "
            "the owning sequences' PagedCache positions instead"
        )


class VarlenPackedCache(Cache):
    """
    Cache for a single fused forward pass over tokens from many sequences.

    Built fresh for each step and thrown away: it owns no state beyond the
    addressing for this one pass, because the durable state is the pool and each
    sequence's own `PagedCache` cursor.
    """

    def __init__(
        self,
        pool,
        num_layers: int,
        block_tables: Sequence[torch.Tensor],
        seq_ids: Sequence[int],
        positions: Sequence[int],
    ):
        if len(seq_ids) != len(positions):
            raise ValueError(
                f"{len(seq_ids)} seq_ids but {len(positions)} positions"
            )
        if not seq_ids:
            raise ValueError("a packed pass needs at least one token")

        self.pool = pool
        self.block_tables = list(block_tables)
        self.seq_ids = list(seq_ids)
        self.positions = list(positions)
        self.num_tokens = len(seq_ids)

        # One flat write index for the whole pass, computed once and reused by
        # every layer. Each token's slot is addressed through its *own*
        # sequence's block table, which is the entire difference from the
        # single-sequence cache.
        self.write_index = self._build_write_index()

        # Through `Cache.__init__` rather than assigning `self.layers`: the base
        # class sets up state of its own (`layer_class_to_replicate` among it)
        # that `update()` dispatch relies on.
        super().__init__(
            layers=[VarlenPackedCacheLayer(self, idx) for idx in range(num_layers)]
        )

    def _build_write_index(self) -> torch.Tensor:
        block_size = self.pool.block_size
        flat: List[int] = []
        for seq_id, position in zip(self.seq_ids, self.positions):
            if seq_id < 0 or seq_id >= len(self.block_tables):
                raise IndexError(
                    f"seq_id {seq_id} has no block table "
                    f"(got {len(self.block_tables)})"
                )
            if position < 0:
                raise ValueError(f"negative position {position}")
            table = self.block_tables[seq_id]
            logical = position // block_size
            if logical >= table.numel():
                raise IndexError(
                    f"position {position} needs logical block {logical} but "
                    f"sequence {seq_id} has a block table of {table.numel()}"
                )
            flat.append(int(table[logical]) * block_size + position % block_size)

        index = torch.tensor(flat, dtype=torch.long, device=self.pool.key_pool.device)
        # Two packed tokens addressing the same slot would make the pass's
        # result depend on scatter order, and would mean one sequence silently
        # overwrote another's KV. Cheap to rule out, and impossible to debug
        # from the output if it ever happened.
        if torch.unique(index).numel() != index.numel():
            raise ValueError(
                "two packed tokens map to the same pool slot; the step's "
                "sequence ids and positions overlap"
            )
        return index

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return 0

    def get_max_cache_shape(self) -> Optional[int]:
        return self.pool.capacity_tokens

    def reset(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self.layers)
