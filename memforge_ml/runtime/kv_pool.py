"""
Paged KV-Cache physical memory pool.

The pool is the single source of truth for every sequence's Key/Value tensors.
It is laid out as a flat array of fixed-size physical blocks; the logical ->
physical mapping lives in the C++ `BlockAllocator`/`Sequence` block tables.

Pool layout: (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
             ^ layer     ^ physical  ^ slot      ^ kv head      ^ feature

Addressing a logical token position `i` of a sequence:
    logical_block = i // block_size
    slot          = i %  block_size
    physical      = block_table[logical_block]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from memforge_ml.kernels import paged_gather


@dataclass(frozen=True)
class ModelKVSpec:
    """Per-token KV geometry derived from a Hugging Face model config."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    device: str

    @classmethod
    def from_config(cls, config, dtype: torch.dtype, device: str) -> "ModelKVSpec":
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        num_kv_heads = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
        return cls(
            num_layers=config.num_hidden_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )

    @property
    def bytes_per_token(self) -> int:
        """Bytes of KV cache consumed by a single token across all layers (K and V)."""
        elem = torch.empty(0, dtype=self.dtype).element_size()
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * elem


class PagedKVPool:
    """Fixed-size physical block pool backing the paged KV cache."""

    def __init__(self, spec: ModelKVSpec, num_blocks: int, block_size: int = 16):
        self.spec = spec
        self.num_blocks = num_blocks
        self.block_size = block_size

        shape = (spec.num_layers, num_blocks, block_size, spec.num_kv_heads, spec.head_dim)
        self.key_pool = torch.zeros(shape, dtype=spec.dtype, device=spec.device)
        self.value_pool = torch.zeros(shape, dtype=spec.dtype, device=spec.device)

    # ---------------------------------------------------------------- sizing

    @property
    def bytes_per_block(self) -> int:
        return self.spec.bytes_per_token * self.block_size

    @property
    def total_bytes(self) -> int:
        return self.bytes_per_block * self.num_blocks

    @property
    def capacity_tokens(self) -> int:
        return self.num_blocks * self.block_size

    @staticmethod
    def blocks_for_budget(spec: ModelKVSpec, block_size: int, budget_mb: float) -> int:
        """Largest number of physical blocks that fits inside a memory budget."""
        per_block = spec.bytes_per_token * block_size
        return max(1, int((budget_mb * 1024 * 1024) // per_block))

    # -------------------------------------------------------------- indexing

    def slot_indices(
        self,
        block_table: torch.Tensor,
        start: int,
        end: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Translate the logical token range [start, end) into (physical_block, slot)
        index tensors. Computed once per step and reused across all layers.
        """
        positions = torch.arange(start, end, device=block_table.device)
        logical = torch.div(positions, self.block_size, rounding_mode="floor")
        slots = positions % self.block_size
        if logical.numel() and int(logical.max()) >= block_table.numel():
            raise IndexError(
                f"logical block {int(logical.max())} is outside a block table of "
                f"{block_table.numel()} entries (token range {start}:{end})"
            )
        return block_table[logical], slots

    @staticmethod
    def block_table_tensor(block_table: List[int], device: str) -> torch.Tensor:
        return torch.tensor(block_table, dtype=torch.long, device=device)

    def flat_indices(
        self,
        block_table: torch.Tensor,
        start: int,
        end: int,
    ) -> torch.Tensor:
        """
        One flat offset per token into the pool viewed as (blocks * slots, ...).

        The form the gather kernel wants: collapsing (block, slot) into a single
        offset turns a two-dimensional advanced index into one index_select,
        which measured a consistent ~1.2x on the decode hot path.
        """
        physical, slots = self.slot_indices(block_table, start, end)
        return paged_gather.flat_slot_index(physical, slots, self.block_size)

    # ----------------------------------------------------------------- I / O

    def write(
        self,
        layer_idx: int,
        physical: torch.Tensor,
        slots: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """
        Scatter `keys`/`values` of shape (1, num_kv_heads, T, head_dim) into the
        physical blocks addressed by (`physical`, `slots`).
        """
        k = keys[0].permute(1, 0, 2).to(self.key_pool.dtype)      # (T, H, D)
        v = values[0].permute(1, 0, 2).to(self.value_pool.dtype)  # (T, H, D)
        self.key_pool[layer_idx, physical, slots] = k
        self.value_pool[layer_idx, physical, slots] = v

    def gather(
        self,
        layer_idx: int,
        physical: torch.Tensor,
        slots: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gather a contiguous (1, num_kv_heads, T, head_dim) view of the KV cache out
        of non-contiguous physical blocks, ready for the attention kernel.

        Kept as the straightforward two-dimensional index so it stays the
        readable reference implementation that `gather_flat` is differentially
        tested against. Callers on the hot path should use `gather_flat`.
        """
        k = self.key_pool[layer_idx, physical, slots]      # (T, H, D)
        v = self.value_pool[layer_idx, physical, slots]    # (T, H, D)
        k = k.permute(1, 0, 2).unsqueeze(0).contiguous()   # (1, H, T, D)
        v = v.permute(1, 0, 2).unsqueeze(0).contiguous()
        return k, v

    def gather_flat(
        self,
        layer_idx: int,
        flat_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Same result as `gather`, via one index_select per tensor."""
        return paged_gather.gather_kv(
            self.key_pool, self.value_pool, layer_idx, flat_index, self.block_size
        )

    def write_flat(
        self,
        layer_idx: int,
        flat_index: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Same effect as `write`, via index_copy_ on the flattened pool."""
        paged_gather.scatter_kv(
            self.key_pool, self.value_pool, layer_idx, flat_index,
            keys, values, self.block_size,
        )

    def zero_(self) -> None:
        self.key_pool.zero_()
        self.value_pool.zero_()

    def describe(self) -> dict:
        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "capacity_tokens": self.capacity_tokens,
            "bytes_per_token": self.spec.bytes_per_token,
            "bytes_per_block": self.bytes_per_block,
            "total_mb": round(self.total_bytes / (1024 * 1024), 2),
            "num_layers": self.spec.num_layers,
            "num_kv_heads": self.spec.num_kv_heads,
            "head_dim": self.spec.head_dim,
            "dtype": str(self.spec.dtype),
            "device": self.spec.device,
        }
