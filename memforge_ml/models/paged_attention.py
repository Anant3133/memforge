"""
PagedAttention Operator: Manages KV cache memory in non-contiguous physical blocks
and performs attention lookups using the C++ logical-to-physical block table.
"""

import math
import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class PagedKVCache:
    """
    Manages physical memory blocks for Key and Value tensors.
    Shape per block: (num_heads, block_size, head_dim)
    """
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Physical memory pool for Key and Value tensors
        self.key_cache = torch.zeros(
            (num_blocks, num_heads, block_size, head_dim),
            dtype=dtype,
            device=device,
        )
        self.value_cache = torch.zeros(
            (num_blocks, num_heads, block_size, head_dim),
            dtype=dtype,
            device=device,
        )

    def write_kv(
        self,
        key_states: torch.Tensor,    # (1, num_heads, seq_len, head_dim)
        value_states: torch.Tensor,  # (1, num_heads, seq_len, head_dim)
        block_table: List[int],
        start_pos: int,
    ) -> None:
        """
        Writes key and value states into physical memory blocks using the block table.
        """
        seq_len = key_states.shape[2]
        for i in range(seq_len):
            logical_pos = start_pos + i
            logical_block_idx = logical_pos // self.block_size
            block_offset = logical_pos % self.block_size

            if logical_block_idx >= len(block_table):
                raise IndexError(f"Logical block {logical_block_idx} exceeds allocated block table length {len(block_table)}")

            physical_block_id = block_table[logical_block_idx]

            self.key_cache[physical_block_id, :, block_offset, :] = key_states[0, :, i, :]
            self.value_cache[physical_block_id, :, block_offset, :] = value_states[0, :, i, :]

    def read_kv(
        self,
        block_table: List[int],
        total_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gathers key and value tensors from physical blocks for attention computation.
        Returns: (num_heads, total_len, head_dim) tensors.
        """
        num_blocks_needed = (total_len + self.block_size - 1) // self.block_size
        gathered_keys = []
        gathered_values = []

        for logical_idx in range(num_blocks_needed):
            physical_block_id = block_table[logical_idx]
            start_in_seq = logical_idx * self.block_size
            end_in_seq = min(start_in_seq + self.block_size, total_len)
            tokens_in_block = end_in_seq - start_in_seq

            k_block = self.key_cache[physical_block_id, :, :tokens_in_block, :]
            v_block = self.value_cache[physical_block_id, :, :tokens_in_block, :]

            gathered_keys.append(k_block)
            gathered_values.append(v_block)

        # Concatenate along token length dimension (dim 1)
        full_keys = torch.cat(gathered_keys, dim=1)    # (num_heads, total_len, head_dim)
        full_values = torch.cat(gathered_values, dim=1) # (num_heads, total_len, head_dim)
        return full_keys, full_values


class PagedAttention(nn.Module):
    """
    Computes Scaled Dot-Product Attention against non-contiguous Paged KV Cache.
    """
    def __init__(self, scale: Optional[float] = None):
        super().__init__()
        self.scale = scale

    def forward(
        self,
        query: torch.Tensor,             # (1, num_heads, query_len, head_dim)
        paged_kv: PagedKVCache,
        block_table: List[int],
        context_len: int,
    ) -> torch.Tensor:
        """
        Executes attention: softmax(Q @ K.T / sqrt(d_k)) @ V
        """
        head_dim = query.shape[-1]
        scale = self.scale or (1.0 / math.sqrt(head_dim))

        # Gather keys & values from paged memory blocks
        keys, values = paged_kv.read_kv(block_table, context_len)
        # keys: (num_heads, context_len, head_dim)
        # values: (num_heads, context_len, head_dim)

        # Reshape query: (num_heads, query_len, head_dim)
        q = query.squeeze(0)

        # Compute attention scores: (num_heads, query_len, context_len)
        attn_scores = torch.matmul(q, keys.transpose(-1, -2)) * scale

        # Apply softmax over context length
        attn_probs = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)

        # Compute output: (num_heads, query_len, head_dim)
        output = torch.matmul(attn_probs, values)

        # Reshape back: (1, query_len, num_heads * head_dim)
        num_heads, query_len, _ = output.shape
        output = output.transpose(0, 1).contiguous().view(1, query_len, num_heads * head_dim)
        return output
