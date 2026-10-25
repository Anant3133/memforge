"""
Batched decode KV assembly (Phase A, tier 1b).

This is the hot spot. Measured on SmolLM2-360M geometry (32 layers, 5 KV heads,
head_dim 64, fp32), the current batched decode path — gather each sequence,
`F.pad` it to the batch maximum, `torch.cat` the batch, per layer — costs:

| batch | context | pad + cat | bytes copied per step |
| ----: | ------: | --------: | --------------------: |
|     4 |     512 |    1.4 ms |                 80 MB |
|     8 |     512 |    3.5 ms |                160 MB |
|     8 |    1024 |   12.9 ms |                320 MB |
|     8 |    2048 |  **46.8 ms** |            **640 MB** |

47 ms and 640 MB *per decode step*, roughly 3× the gather it sits on top of.
The Phase A plan ranks the fused gather first and this second; the measurement
says the reverse, which is recorded in
[AKSHIT_WORKPLAN_AND_CODEBASE_AUDIT.md](../../AKSHIT_WORKPLAN_AND_CODEBASE_AUDIT.md)
§A.2.4.

The fix is to stop making three copies. Today each sequence's KV is copied once
by the gather, again by `F.pad`, and a third time by `torch.cat`. Here the batch
tensor is allocated once and each sequence is gathered **directly into its own
left-aligned window of it**, so padding is "the region nobody wrote" rather than
a copy, and concatenation does not happen because the destination was always a
single tensor.

Buffers are cached by shape and reused across steps, because contexts grow by
one token per step and reallocating a 640 MB pair every step is most of the cost
this module exists to remove.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

from memforge_ml.kernels.paged_gather import flat_slot_index


class BatchedDecodeBuffers:
    """
    Reusable `(batch, heads, padded_len, head_dim)` scratch for batched decode.

    Keyed by `(batch, padded_len)`, with `padded_len` rounded up to a multiple of
    `pad_to` so a context creeping from 1024 to 1088 reuses one buffer instead of
    allocating 64 new pairs. One entry per shape per layer-stack, holding K and V.

    Zeroing matters: the padded region is read by the model, masked out by
    `attention_mask`, and masking is multiplicative — so whatever is in those
    slots must be finite. Reusing a dirty buffer would leave another sequence's
    keys there, and while the mask makes them not *count*, NaN or inf would
    still poison the softmax. The buffer is therefore zeroed on allocation and
    the padding region re-zeroed on reuse.
    """

    def __init__(self, num_layers: int, pad_to: int = 64):
        self.num_layers = num_layers
        self.pad_to = max(1, pad_to)
        self._cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    def padded_len(self, max_len: int) -> int:
        return ((max_len + self.pad_to - 1) // self.pad_to) * self.pad_to

    def acquire(
        self,
        batch: int,
        max_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Return `(keys, values, padded_len)` buffers of
        `(num_layers, batch, heads, padded_len, head_dim)`, zeroed and ready.
        """
        padded = self.padded_len(max_len)
        key = (batch, padded)
        entry = self._cache.get(key)
        if entry is None:
            shape = (self.num_layers, batch, num_heads, padded, head_dim)
            entry = (
                torch.zeros(shape, dtype=dtype, device=device),
                torch.zeros(shape, dtype=dtype, device=device),
            )
            self._cache[key] = entry
        else:
            # Reused: clear it so no stale keys from a previous step survive in
            # the region this step will treat as padding.
            entry[0].zero_()
            entry[1].zero_()
        return entry[0], entry[1], padded

    def clear(self) -> None:
        self._cache.clear()

    @property
    def num_cached_shapes(self) -> int:
        return len(self._cache)


def assemble_batch(
    pool,
    buffers: BatchedDecodeBuffers,
    block_tables: Sequence[torch.Tensor],
    context_lengths: Sequence[int],
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor, int]:
    """
    Gather a decode batch straight into one preallocated, left-padded tensor.

    Left padding (not right) because the model is given explicit `position_ids`
    and an `attention_mask`, and the existing engine convention places each
    sequence's real tokens at the *end* of its row so the newest token is
    adjacent to the one being generated. This keeps that convention.

    Returns `(per_layer_kv, attention_mask, padded_len)`, where `per_layer_kv` is
    a list of `(keys, values)` views -- one pair per layer, each
    `(batch, heads, padded_len, head_dim)` -- suitable for seeding a cache, and
    `attention_mask` is `(batch, padded_len)` with 1 on real tokens.

    No `F.pad`, no `torch.cat`: the only copy per sequence per layer is the
    gather itself.
    """
    batch = len(context_lengths)
    if batch == 0:
        raise ValueError("assemble_batch requires at least one sequence")
    if len(block_tables) != batch:
        raise ValueError(
            f"{len(block_tables)} block tables for {batch} sequences"
        )

    max_len = max(context_lengths)
    num_heads, head_dim = pool.spec.num_kv_heads, pool.spec.head_dim
    keys, values, padded = buffers.acquire(
        batch, max_len, num_heads, head_dim, pool.key_pool.dtype, pool.spec.device
    )

    flat_key = pool.key_pool.view(pool.spec.num_layers, -1, num_heads, head_dim)
    flat_value = pool.value_pool.view(pool.spec.num_layers, -1, num_heads, head_dim)

    # Index translation is shared across all layers: it depends only on the
    # block table and the context length, neither of which varies by layer.
    # Measured at 0.049 ms for 2048 tokens, so this is for tidiness, not speed.
    flat_indices: List[torch.Tensor] = []
    for table, ctx in zip(block_tables, context_lengths):
        physical, slots = pool.slot_indices(table, 0, ctx)
        flat_indices.append(flat_slot_index(physical, slots, pool.block_size))

    for layer_idx in range(pool.spec.num_layers):
        src_k, src_v = flat_key[layer_idx], flat_value[layer_idx]
        dst_k, dst_v = keys[layer_idx], values[layer_idx]
        for row, (index, ctx) in enumerate(zip(flat_indices, context_lengths)):
            start = padded - ctx  # left-aligned to the end of the row
            # index_select gives (ctx, heads, dim); the destination window is
            # (heads, ctx, dim), so one transposed copy lands it in place. That
            # is a single copy per sequence per layer, where the old path made
            # three.
            gathered = torch.index_select(src_k, 0, index)
            dst_k[row, :, start:padded, :] = gathered.permute(1, 0, 2)
            gathered = torch.index_select(src_v, 0, index)
            dst_v[row, :, start:padded, :] = gathered.permute(1, 0, 2)

    attention_mask = torch.zeros(
        (batch, padded), dtype=torch.long, device=pool.spec.device
    )
    for row, ctx in enumerate(context_lengths):
        attention_mask[row, padded - ctx:] = 1

    per_layer = [(keys[i], values[i]) for i in range(pool.spec.num_layers)]
    return per_layer, attention_mask, padded


def scatter_new_tokens(
    pool,
    block_tables: Sequence[torch.Tensor],
    context_lengths: Sequence[int],
    new_keys: Sequence[torch.Tensor],
    new_values: Sequence[torch.Tensor],
) -> None:
    """
    Write each sequence's newly produced K/V back into its own physical blocks.

    `new_keys[layer]` is `(batch, heads, 1, head_dim)` — the single position the
    forward pass just produced. Each row goes to logical position
    `context_lengths[row]` of its own sequence, which is what keeps the pool the
    source of truth rather than the batch buffer.
    """
    for layer_idx in range(pool.spec.num_layers):
        layer_k, layer_v = new_keys[layer_idx], new_values[layer_idx]
        for row, (table, ctx) in enumerate(zip(block_tables, context_lengths)):
            physical, slots = pool.slot_indices(table, ctx, ctx + 1)
            pool.write(
                layer_idx,
                physical,
                slots,
                layer_k[row : row + 1],
                layer_v[row : row + 1],
            )
