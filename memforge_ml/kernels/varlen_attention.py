"""
Variable-length block-wise paged attention (Phase C, prefill/decode fusion).

`block_attention.py` computes attention for *one* query position against one
sequence's blocks. That is the decode case, and it is why the engine still runs
a prefill chunk and a decode step as two separate forward passes: the second
bullet of Phase C. The cost of that split is measurable and it is the reason
the chunked-prefill stall ratio floors at ~2.5x instead of approaching 1x --
a step carrying a chunk pays for two passes through the whole model.

Fusing them needs attention over a *packed* batch: one flat run of query
tokens belonging to different sequences, each with its own block table, its own
context length and its own absolute position. There is no shared `kv_len`, so
the standard `(batch, 1, q_len, kv_len)` mask cannot express it at all -- which
is why this has to be a kernel rather than a mask change.

**The shape of the problem.** Given query tokens packed as
`(1, H, T_total, D)`, plus for each token the sequence it belongs to and its
absolute position in that sequence, every token attends to its own sequence's
keys at positions `0..p` inclusive. So:

  * a decode token has `p = context_length - 1` and sees everything;
  * a prefill chunk's tokens have consecutive `p`, each seeing the chunk's
    earlier tokens and everything before the chunk, and nothing after -- which
    is ordinary causality, expressed per token rather than per batch row.

Both fall out of the same comparison, `key_position <= query_position`, applied
inside the block loop. There is no separate prefill and decode case here, and
that is the point: it is what lets one pass serve both.

**Working memory** is `O(rows_of_one_sequence * block_size)`, not
`O(T_total * max_context)`. The online-softmax recurrence is the same identity
`block_attention.py` documents, carried per (head, row) instead of per head.

Like its single-query sibling this is a reference implementation and is slower
than the materialised path on CPU, for the same reason: Python-level iteration
over blocks pays interpreter overhead the vectorised gather avoids. Its value
is that it is the only formulation that *can* fuse the two step types, and that
it is provably equal to per-sequence causal attention -- `tests/python/
test_varlen_attention.py` asserts that to within fp32 tolerance against a
reference that shares no code with it.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch

#: Score value for a masked key. `-inf` is correct and intended -- it makes the
#: key contribute exactly zero probability -- but it needs care, because a query
#: token early in a prefill chunk sees *no* key in the chunk's later blocks, so
#: its whole score row there is `-inf`. `exp(-inf - -inf)` is `nan`, so the
#: recurrence below guards both rescale factors on `isfinite` and multiplies the
#: probabilities by the visibility mask rather than relying on `exp` alone.
MASK_FILL = float("-inf")


def group_by_sequence(seq_ids: Sequence[int]) -> List[Tuple[int, List[int]]]:
    """
    Which packed rows belong to which sequence, preserving first-seen order.

    Returned as a list rather than a dict so the iteration order is explicit and
    reproducible; attention is order-independent but a differential test against
    a reference is much easier to read when the rows come back in a fixed order.
    """
    order: List[int] = []
    rows: dict = {}
    for row, seq_id in enumerate(seq_ids):
        if seq_id not in rows:
            rows[seq_id] = []
            order.append(seq_id)
        rows[seq_id].append(row)
    return [(seq_id, rows[seq_id]) for seq_id in order]


def paged_attention_varlen(
    query: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    layer_idx: int,
    block_tables: Sequence[torch.Tensor],
    seq_ids: Sequence[int],
    positions: Sequence[int],
    block_size: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Attention for a packed run of query tokens from several sequences.

    Args:
        query: `(1, num_heads, T_total, D)`, all sequences' query tokens packed
            along the token axis in any order. `num_heads` may exceed the pool's
            `num_kv_heads`, in which case grouped-query attention applies.
        key_pool, value_pool: `(layers, blocks, slots, kv_heads, head_dim)`.
        block_tables: one block table per *distinct* value in `seq_ids`, indexed
            by that value. Entry `i` is the table of the sequence whose id is
            `i`, so callers pass a list positionally rather than a dict.
        seq_ids: `T_total` entries -- which sequence each query token belongs
            to, as an index into `block_tables`.
        positions: `T_total` entries -- each query token's absolute position in
            its own sequence. This is what supplies causality; it is not
            derived from the packing order, because the packing order has no
            meaning across sequences.
        scale: defaults to `1/sqrt(head_dim)`.

    Returns:
        `(1, num_heads, T_total, D)`, rows in the same order as the input.
    """
    if query.shape[0] != 1:
        raise ValueError(f"expected a packed batch of 1, got {tuple(query.shape)}")
    total = query.shape[2]
    if len(seq_ids) != total or len(positions) != total:
        raise ValueError(
            f"{total} query tokens but {len(seq_ids)} seq_ids and "
            f"{len(positions)} positions"
        )
    if total == 0:
        raise ValueError("paged_attention_varlen requires at least one query token")
    if any(p < 0 for p in positions):
        raise ValueError(f"negative query position in {list(positions)}")

    num_heads, head_dim = query.shape[1], query.shape[3]
    num_kv_heads = key_pool.shape[3]
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"{num_heads} query heads is not a multiple of {num_kv_heads} kv heads"
        )
    group = num_heads // num_kv_heads
    scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)

    dtype = torch.float32  # accumulate in fp32 regardless of pool dtype
    device = query.device
    out = torch.zeros((num_heads, total, head_dim), dtype=dtype, device=device)

    for seq_id, rows in group_by_sequence(seq_ids):
        if seq_id < 0 or seq_id >= len(block_tables):
            raise IndexError(
                f"seq_id {seq_id} has no block table (got {len(block_tables)})"
            )
        block_table = block_tables[seq_id]
        row_index = torch.tensor(rows, dtype=torch.long, device=device)
        row_positions = torch.tensor(
            [positions[r] for r in rows], dtype=torch.long, device=device
        )

        # The furthest key any of this sequence's rows may read. Blocks past it
        # are not visited at all, which is what keeps a decode token in the same
        # pass as a long prefill chunk from paying for the chunk's blocks.
        last_key = int(row_positions.max())
        num_logical = (last_key + 1 + block_size - 1) // block_size
        if num_logical > block_table.numel():
            raise IndexError(
                f"sequence {seq_id} reads up to position {last_key}, needing "
                f"{num_logical} blocks, but its block table has "
                f"{block_table.numel()} entries"
            )

        q = query[0, :, row_index, :].to(dtype)  # (H, R, D)
        rows_n = len(rows)

        running_max = torch.full(
            (num_heads, rows_n), float("-inf"), dtype=dtype, device=device
        )
        running_sum = torch.zeros((num_heads, rows_n), dtype=dtype, device=device)
        accumulator = torch.zeros(
            (num_heads, rows_n, head_dim), dtype=dtype, device=device
        )

        for logical in range(num_logical):
            physical = int(block_table[logical])
            base = logical * block_size
            # Trim the last visited block to the tokens that exist. This is a
            # *flop saving*, not a correctness requirement: the causal mask
            # below already excludes every position past `last_key`, and
            # `masked_fill` overwrites whatever garbage an unwritten slot held,
            # including nan. Mutation testing confirmed that widening this
            # changes no output -- so it is kept for the work it avoids, which
            # matters most in exactly the case this kernel exists for, a single
            # decode row sharing a pass with a long prefill chunk.
            valid = min(block_size, last_key + 1 - base)

            k_block = key_pool[layer_idx, physical, :valid].to(dtype)   # (valid,Hkv,D)
            v_block = value_pool[layer_idx, physical, :valid].to(dtype)
            if group > 1:
                k_block = k_block.repeat_interleave(group, dim=1)
                v_block = v_block.repeat_interleave(group, dim=1)

            # (H, R, valid)
            scores = torch.einsum("hrd,thd->hrt", q, k_block) * scale

            # Causality, per token rather than per batch row: a key is visible
            # only to query tokens at or after its own position. This single
            # comparison is what makes prefill and decode the same case.
            key_positions = torch.arange(base, base + valid, device=device)
            visible = key_positions.unsqueeze(0) <= row_positions.unsqueeze(1)
            # No emptiness check: `last_key` is the greatest row position, so
            # the row holding it sees `base` in every visited block and at least
            # one entry is always visible. Guarding that would be unreachable.
            scores = scores.masked_fill(~visible.unsqueeze(0), MASK_FILL)

            block_max = scores.max(dim=-1).values                        # (H, R)
            new_max = torch.maximum(running_max, block_max)
            # A row that sees nothing in this block has block_max == -inf. If it
            # has also seen nothing so far, new_max stays -inf and
            # exp(-inf - -inf) would be nan, so both rescale factors are
            # guarded on isfinite and the running max is left untouched.
            finite = torch.isfinite(new_max)
            safe_max = torch.where(finite, new_max, torch.zeros_like(new_max))
            rescale_old = torch.where(
                torch.isfinite(running_max),
                torch.exp(running_max - safe_max),
                torch.zeros_like(running_max),
            )
            # Masked entries contribute exactly zero without further help:
            # `safe_max` is always finite, and `exp(-inf - finite)` is 0.0
            # exactly in IEEE arithmetic. Multiplying by the visibility mask
            # again would be a no-op, which mutation testing confirmed.
            probs = torch.exp(scores - safe_max.unsqueeze(-1))

            running_sum = running_sum * rescale_old + probs.sum(dim=-1)
            accumulator = accumulator * rescale_old.unsqueeze(-1) + torch.einsum(
                "hrt,thd->hrd", probs, v_block
            )
            running_max = torch.where(finite, new_max, running_max)

        if not bool((running_sum > 0).all()):
            raise ValueError(
                f"sequence {seq_id} has a query token with no visible keys; "
                f"positions {row_positions.tolist()} cannot all be served"
            )
        out[:, row_index, :] = accumulator / running_sum.unsqueeze(-1)

    return out.unsqueeze(0).to(query.dtype)


def reference_varlen_attention(
    query: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    layer_idx: int,
    block_tables: Sequence[torch.Tensor],
    seq_ids: Sequence[int],
    positions: Sequence[int],
    block_size: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    What `paged_attention_varlen` must equal, computed the obvious way.

    Materialises each query token's whole visible context and runs one plain
    softmax over it. Deliberately shares no code with the kernel: it indexes
    the pool directly, token by token, and never touches the online-softmax
    recurrence. A reference that reused the block loop would agree with a
    broken block loop.
    """
    num_heads, head_dim = query.shape[1], query.shape[3]
    num_kv_heads = key_pool.shape[3]
    group = num_heads // num_kv_heads
    scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
    dtype = torch.float32
    total = query.shape[2]

    out = torch.zeros((num_heads, total, head_dim), dtype=dtype, device=query.device)
    for row in range(total):
        table = block_tables[seq_ids[row]]
        visible = positions[row] + 1

        keys = torch.empty((visible, num_kv_heads, head_dim), dtype=dtype,
                           device=query.device)
        values = torch.empty_like(keys)
        for pos in range(visible):
            physical = int(table[pos // block_size])
            slot = pos % block_size
            keys[pos] = key_pool[layer_idx, physical, slot].to(dtype)
            values[pos] = value_pool[layer_idx, physical, slot].to(dtype)

        if group > 1:
            keys = keys.repeat_interleave(group, dim=1)
            values = values.repeat_interleave(group, dim=1)

        q = query[0, :, row, :].to(dtype)                       # (H, D)
        scores = torch.einsum("hd,thd->ht", q, keys) * scale    # (H, visible)
        weights = torch.softmax(scores, dim=-1)
        out[:, row, :] = torch.einsum("ht,thd->hd", weights, values)

    return out.unsqueeze(0).to(query.dtype)


def paged_attention_varlen_materialised(
    query: torch.Tensor,
    pool,
    layer_idx: int,
    block_tables: Sequence[torch.Tensor],
    seq_ids: Sequence[int],
    positions: Sequence[int],
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    The same function as `paged_attention_varlen`, computed the fast way.

    `paged_attention_varlen` keeps working memory at `O(rows * block_size)` by
    walking one block at a time, which is the shape a GPU kernel wants and is
    the only version whose memory does not grow with context. On CPU that loop
    is the dominant cost: a 512-token context at block 16 is 32 Python
    iterations *per layer*, each doing tiny tensor ops, and the interpreter
    overhead swamps the arithmetic. Measured, it made fused steps so much slower
    per pass that removing a third of the forward passes still left the victim's
    worst stall twice as bad -- the pass saving was real and the per-pass cost
    more than cancelled it.

    So this exists: gather each sequence's context once with a single
    `index_select`, then one masked softmax over it. Three tensor operations per
    sequence per layer instead of thirty-two. Identical output to within fp32
    reduction order, which `test_varlen_attention.py` asserts differentially.

    The trade is memory: scores are `O(heads * rows * context)` and the gathered
    context is `O(context)`, so this gives up the bounded-working-memory
    property that makes the block-wise version worth keeping. Both are kept for
    the same reason tiers 1 and 3 of Phase A are both kept -- one is fast here,
    the other is the structure that pays off on a device that can hide the
    per-block overhead.
    """
    if query.shape[0] != 1:
        raise ValueError(f"expected a packed batch of 1, got {tuple(query.shape)}")
    total = query.shape[2]
    if len(seq_ids) != total or len(positions) != total:
        raise ValueError(
            f"{total} query tokens but {len(seq_ids)} seq_ids and "
            f"{len(positions)} positions"
        )
    if total == 0:
        raise ValueError("paged_attention_varlen requires at least one query token")
    if any(p < 0 for p in positions):
        raise ValueError(f"negative query position in {list(positions)}")

    num_heads, head_dim = query.shape[1], query.shape[3]
    num_kv_heads = pool.key_pool.shape[3]
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"{num_heads} query heads is not a multiple of {num_kv_heads} kv heads"
        )
    group = num_heads // num_kv_heads
    scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)

    dtype = torch.float32
    device = query.device
    out = torch.zeros((num_heads, total, head_dim), dtype=dtype, device=device)

    for seq_id, rows in group_by_sequence(seq_ids):
        if seq_id < 0 or seq_id >= len(block_tables):
            raise IndexError(
                f"seq_id {seq_id} has no block table (got {len(block_tables)})"
            )
        block_table = block_tables[seq_id]
        row_index = torch.tensor(rows, dtype=torch.long, device=device)
        row_positions = torch.tensor(
            [positions[r] for r in rows], dtype=torch.long, device=device
        )
        context_length = int(row_positions.max()) + 1

        # Raised here rather than left to the gather, so the message names the
        # sequence: a silent read past the table would land in another
        # sequence's block.
        needed = (context_length + pool.block_size - 1) // pool.block_size
        if needed > block_table.numel():
            raise IndexError(
                f"sequence {seq_id} reads up to position {context_length - 1}, "
                f"needing {needed} blocks, but its block table has "
                f"{block_table.numel()} entries"
            )

        flat = pool.flat_indices(block_table, 0, context_length)
        keys, values = pool.gather_flat(layer_idx, flat)   # (1, Hkv, ctx, D)
        keys = keys[0].to(dtype)
        values = values[0].to(dtype)
        if group > 1:
            keys = keys.repeat_interleave(group, dim=0)
            values = values.repeat_interleave(group, dim=0)

        q = query[0, :, row_index, :].to(dtype)                      # (H, R, D)
        scores = torch.einsum("hrd,htd->hrt", q, keys) * scale       # (H, R, ctx)

        # The same per-token causality as the block-wise version: a key is
        # visible only at or after its own position. Unwritten slots cannot
        # appear here at all, because the gather stops at `context_length`.
        key_positions = torch.arange(context_length, device=device)
        visible = key_positions.unsqueeze(0) <= row_positions.unsqueeze(1)
        scores = scores.masked_fill(~visible.unsqueeze(0), MASK_FILL)

        weights = torch.softmax(scores, dim=-1)
        out[:, row_index, :] = torch.einsum("hrt,htd->hrd", weights, values)

    return out.unsqueeze(0).to(query.dtype)
