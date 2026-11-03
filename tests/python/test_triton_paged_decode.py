"""Triton paged decode kernel vs a plain softmax reference."""
import pytest
import torch

from memforge_ml.kernels import triton_paged_decode as K

pytestmark = pytest.mark.skipif(not K.available(), reason="needs triton + CUDA")


def _reference(q, kflat, vflat, tables, ctx, bs, scale):
    B, Hq, D = q.shape
    Hkv = kflat.shape[1]
    G = Hq // Hkv
    outs = []
    for b in range(B):
        n = int(ctx[b])
        pos = torch.arange(n, device=q.device)
        slots = tables[b][pos // bs] * bs + pos % bs
        k = kflat[slots].float()              # [n, Hkv, D]
        v = vflat[slots].float()
        qb = q[b].float().view(Hkv, G, D)
        s = torch.einsum("hgd,nhd->hgn", qb, k) * scale
        p = torch.softmax(s, -1)
        outs.append(torch.einsum("hgn,nhd->hgd", p, v).reshape(Hq, D))
    return torch.stack(outs)


def _case(B, Hq, Hkv, D, bs, ctxs, dtype, splits=8, tile=64, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    nblocks = 4 + sum((c + bs - 1) // bs for c in ctxs) + 8
    kflat = torch.randn(nblocks * bs, Hkv, D, device="cuda", dtype=dtype, generator=g)
    vflat = torch.randn_like(kflat)
    q = torch.randn(B, Hq, D, device="cuda", dtype=dtype, generator=g)
    maxb = max((c + bs - 1) // bs for c in ctxs) + 2
    perm = torch.randperm(nblocks, device="cuda", generator=g)  # scattered, non-contiguous
    tables = torch.zeros(B, maxb, dtype=torch.long, device="cuda")
    i = 0
    for b, c in enumerate(ctxs):
        n = (c + bs - 1) // bs
        tables[b, :n] = perm[i:i + n]
        i += n
    ctx = torch.tensor(ctxs, device="cuda", dtype=torch.long)
    scale = D ** -0.5
    out = K.paged_decode_attention(q, kflat, vflat, tables, ctx, bs, max(ctxs), scale,
                                   num_splits=splits, tile=tile)
    ref = _reference(q, kflat, vflat, tables, ctx, bs, scale)
    return out.float(), ref


@pytest.mark.parametrize("dtype,tol", [(torch.float16, 2e-3), (torch.bfloat16, 2e-2)])
@pytest.mark.parametrize("Hq,Hkv", [(8, 8), (16, 4), (28, 4), (32, 8)])
def test_matches_reference(Hq, Hkv, dtype, tol):
    out, ref = _case(3, Hq, Hkv, 128, 16, [1, 77, 300], dtype)
    assert (out - ref).abs().max().item() < tol


def test_ragged_lengths_and_tile_boundaries():
    for ctxs in ([1], [63, 64, 65], [2048, 5], [129, 128, 127, 16]):
        out, ref = _case(len(ctxs), 16, 4, 128, 16, ctxs, torch.float16)
        assert (out - ref).abs().max().item() < 2e-3, ctxs


def test_single_split_and_many_splits_agree():
    a, ref = _case(2, 16, 4, 128, 16, [500, 33], torch.float16, splits=1)
    b, ref_b = _case(2, 16, 4, 128, 16, [500, 33], torch.float16, splits=16)
    assert (a - ref).abs().max().item() < 2e-3
    assert (b - ref_b).abs().max().item() < 2e-3


def test_ignores_garbage_beyond_context():
    """Slots past ctx_len (stale pool data / padding) must have no influence."""
    torch.manual_seed(1)
    bs = 16
    kflat = torch.randn(64 * bs, 4, 128, device="cuda", dtype=torch.float16)
    vflat = torch.randn_like(kflat)
    q = torch.randn(1, 16, 128, device="cuda", dtype=torch.float16)
    tbl = torch.arange(8, device="cuda").view(1, 8)
    ctx = torch.tensor([40], device="cuda")
    a = K.paged_decode_attention(q, kflat, vflat, tbl, ctx, bs, 128, 128 ** -0.5)
    kflat[40:128] = 1e4
    vflat[40:128] = -1e4
    b = K.paged_decode_attention(q, kflat, vflat, tbl, ctx, bs, 128, 128 ** -0.5)
    assert torch.equal(a, b)


def test_context_length_is_a_runtime_value_in_a_cuda_graph():
    """One captured graph, replayed with different lengths, stays correct."""
    bs, D = 16, 128
    kflat = torch.randn(64 * bs, 4, D, device="cuda", dtype=torch.float16)
    vflat = torch.randn_like(kflat)
    q = torch.randn(2, 16, D, device="cuda", dtype=torch.float16)
    tbl = torch.arange(16, device="cuda").repeat(2, 1)
    ctx = torch.tensor([10, 200], device="cuda")
    out = torch.empty(2, 16, D, device="cuda", dtype=torch.float16)
    K.paged_decode_attention(q, kflat, vflat, tbl, ctx, bs, 256, D ** -0.5, out=out)  # warm/compile
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        K.paged_decode_attention(q, kflat, vflat, tbl, ctx, bs, 256, D ** -0.5, out=out)
    for lens in ([10, 200], [255, 1], [64, 64]):
        ctx.copy_(torch.tensor(lens))
        g.replay()
        ref = _reference(q, kflat, vflat, tbl, ctx, bs, D ** -0.5)
        assert (out.float() - ref).abs().max().item() < 2e-3, lens
