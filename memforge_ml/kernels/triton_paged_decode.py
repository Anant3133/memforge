"""
Triton paged-decode attention (backlog B1): attention reads K/V straight out of
the block pool through the block table, so the per-step gather copy disappears.

Why. The CUDA-graphed paged step (`graphed_paged_decode.py`) removed launch
overhead but still copies the whole context (`pool[gather_idx]`) every layer
every step. At 2048 tokens on a 7B/8B model that copy is what keeps paged decode
at 0.91-0.93x of contiguous.

Shape of the kernel (flash-decoding):
  * grid = (batch, kv_head, split). Each program owns a contiguous range of
    tokens of one sequence and one KV head, and handles *all* query heads that
    share that KV head (GQA), so each K/V byte is read once.
  * stage 1 writes a partial (max, sum, accumulator) per split;
  * stage 2 merges splits with the log-sum-exp identity.
  * `num_splits` and every loop bound are compile-time constants; the per-row
    context length is a *device tensor*, so the whole thing is CUDA-graph safe.

fp32 accumulation, output cast to the query dtype.
"""

from __future__ import annotations

import torch

try:  # Triton is optional; callers check `available()`.
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # pragma: no cover - environment without triton
    _HAVE_TRITON = False


def available() -> bool:
    return _HAVE_TRITON and torch.cuda.is_available()


if _HAVE_TRITON:

    @triton.jit
    def _stage1(
        Q, KP, VP, TBL, CTX, PART_M, PART_L, PART_O,
        scale,
        stride_qb, stride_qh,
        stride_tb,
        stride_ks, stride_kh,           # K pool flat [slots, Hkv, D]
        stride_pm_b, stride_pm_h,       # PART_M/L [B, Hq, S]
        stride_po_b, stride_po_h, stride_po_s,  # PART_O [B, Hq, S, D]
        BLOCK_SIZE: tl.constexpr,
        G: tl.constexpr,                # query heads per kv head
        GP: tl.constexpr,               # G padded to >= 16
        D: tl.constexpr,
        TILE: tl.constexpr,             # tokens per inner iteration
        SPLITS: tl.constexpr,
        TOKENS_PER_SPLIT: tl.constexpr,
    ):
        b = tl.program_id(0)
        hk = tl.program_id(1)
        sp = tl.program_id(2)

        ctx = tl.load(CTX + b)  # number of valid tokens (incl. the new one)
        start = sp * TOKENS_PER_SPLIT

        g = tl.arange(0, GP)
        d = tl.arange(0, D)
        gmask = g < G
        hq = hk * G + g
        q = tl.load(
            Q + b * stride_qb + hq[:, None] * stride_qh + d[None, :],
            mask=gmask[:, None], other=0.0,
        )  # [GP, D]

        m_i = tl.full([GP], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([GP], dtype=tl.float32)
        acc = tl.zeros([GP, D], dtype=tl.float32)

        for t0 in range(0, TOKENS_PER_SPLIT, TILE):
            tok = start + t0 + tl.arange(0, TILE)
            valid = tok < ctx
            blk = tl.load(TBL + b * stride_tb + tok // BLOCK_SIZE, mask=valid, other=0)
            slot = blk * BLOCK_SIZE + tok % BLOCK_SIZE
            kptr = KP + slot[:, None] * stride_ks + hk * stride_kh + d[None, :]
            k = tl.load(kptr, mask=valid[:, None], other=0.0)          # [TILE, D]
            s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale                          # [GP, TILE]
            s = tl.where(valid[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp(s - m_safe[:, None])
            alpha = tl.exp(tl.where(m_i == float("-inf"), float("-inf"), m_i) - m_safe)
            l_i = l_i * alpha + tl.sum(p, 1)
            vptr = VP + slot[:, None] * stride_ks + hk * stride_kh + d[None, :]
            v = tl.load(vptr, mask=valid[:, None], other=0.0)           # [TILE, D]
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee")
            m_i = m_new

        tl.store(PART_M + b * stride_pm_b + hq * stride_pm_h + sp, m_i, mask=gmask)
        tl.store(PART_L + b * stride_pm_b + hq * stride_pm_h + sp, l_i, mask=gmask)
        tl.store(
            PART_O + b * stride_po_b + hq[:, None] * stride_po_h + sp * stride_po_s + d[None, :],
            acc, mask=gmask[:, None],
        )

    @triton.jit
    def _stage2(
        PART_M, PART_L, PART_O, OUT,
        stride_pm_b, stride_pm_h,
        stride_po_b, stride_po_h, stride_po_s,
        stride_ob, stride_oh,
        D: tl.constexpr,
        SPLITS: tl.constexpr,
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)
        s = tl.arange(0, SPLITS)
        d = tl.arange(0, D)
        m = tl.load(PART_M + b * stride_pm_b + h * stride_pm_h + s)
        l = tl.load(PART_L + b * stride_pm_b + h * stride_pm_h + s)
        m_all = tl.max(m, 0)
        w = tl.where(m == float("-inf"), 0.0, tl.exp(m - m_all))
        denom = tl.sum(l * w, 0)
        o = tl.load(
            PART_O + b * stride_po_b + h * stride_po_h + s[:, None] * stride_po_s + d[None, :]
        )  # [SPLITS, D]
        num = tl.sum(o * w[:, None], 0)
        tl.store(OUT + b * stride_ob + h * stride_oh + d, (num / denom).to(OUT.dtype.element_ty))


def paged_decode_attention(
    q: torch.Tensor,            # [B, Hq, D]
    key_flat: torch.Tensor,     # [slots, Hkv, D]  (one layer, pool viewed flat)
    value_flat: torch.Tensor,   # [slots, Hkv, D]
    block_tables: torch.Tensor,  # [B, max_blocks] int
    ctx_lens: torch.Tensor,     # [B] int: valid tokens incl. the new one
    block_size: int,
    max_ctx: int,               # compile-time upper bound on ctx for this graph
    scale: float,
    num_splits: int = 8,
    tile: int = 64,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single-token attention for each of B sequences over a paged cache."""
    if not available():
        raise RuntimeError("triton paged decode needs triton and CUDA")
    B, Hq, D = q.shape
    Hkv = key_flat.shape[1]
    if Hq % Hkv:
        raise ValueError("query heads must be a multiple of kv heads")
    G = Hq // Hkv
    GP = max(16, triton.next_power_of_2(G))
    splits = max(1, triton.next_power_of_2(num_splits))
    per = triton.cdiv(max(max_ctx, 1), splits)
    per = triton.cdiv(per, tile) * tile
    dev = q.device
    pm = torch.empty((B, Hq, splits), device=dev, dtype=torch.float32)
    pl = torch.empty_like(pm)
    po = torch.empty((B, Hq, splits, D), device=dev, dtype=torch.float32)
    if out is None:
        out = torch.empty((B, Hq, D), device=dev, dtype=q.dtype)
    _stage1[(B, Hkv, splits)](
        q, key_flat, value_flat, block_tables, ctx_lens, pm, pl, po, scale,
        q.stride(0), q.stride(1), block_tables.stride(0),
        key_flat.stride(0), key_flat.stride(1),
        pm.stride(0), pm.stride(1),
        po.stride(0), po.stride(1), po.stride(2),
        BLOCK_SIZE=block_size, G=G, GP=GP, D=D, TILE=tile,
        SPLITS=splits, TOKENS_PER_SPLIT=per,
    )
    _stage2[(B, Hq)](
        pm, pl, po, out,
        pm.stride(0), pm.stride(1),
        po.stride(0), po.stride(1), po.stride(2),
        out.stride(0), out.stride(1),
        D=D, SPLITS=splits,
    )
    return out
