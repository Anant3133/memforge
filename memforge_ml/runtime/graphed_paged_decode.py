"""
CUDA-graphed paged decode step.

Why this exists. On the A100 the paged decode path ran at 0.85-0.94x of stock
contiguous decoding (Phase A target: >= 0.95x). The cost is not memory
bandwidth: it is launch overhead. Per layer the paged path adds an index
translation, a K write, a V write, a K gather and a V gather, each its own
kernel launch from Python, on a step that is already launch-bound at batch 1.

Here the whole step -- index translation, the write of the new token's K/V
into its physical block, the gather of the context from the pool, the causal
mask and the model forward -- is one captured CUDA graph. Everything that
varies between steps (tokens, positions, block tables) is a *device tensor
input*; nothing host-derived is baked in at capture time (the lesson of
`GraphedDraftPool`).

The pool stays the single source of truth: the new K/V is written straight
into its own physical block inside the graph, so no scatter-back pass follows.

Shapes are bucketed so the number of graphs stays small: batch is padded up to
a power of two by *repeating row 0* (a duplicate row recomputes and rewrites
identical values to the same slot, so it can corrupt nothing), and the gathered
context length is padded up to a multiple of `ctx_bucket` tokens, with the
padding masked out.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch


def _make_layer_class():
    from transformers.cache_utils import CacheLayerMixin

    class PagedGraphLayer(CacheLayerMixin):
        is_sliding = False
        is_compileable = False

        def __init__(self, owner: "GraphedPagedDecoder", layer_idx: int):
            super().__init__()
            self.owner, self.layer_idx = owner, layer_idx
            self.is_initialized = True

        def update(self, key_states, value_states, *args, **kwargs):
            o, i = self.owner, self.layer_idx
            b, P = o._b, o._P
            widx = o._widx[:b]
            # (B, H, 1, D) -> (B, H, D), written at each row's physical slot.
            o._flat_k[i][widx] = key_states[:, :, 0]
            o._flat_v[i][widx] = value_states[:, :, 0]
            g = o._gidx[:b, :P]
            return (
                o._flat_k[i][g].permute(0, 2, 1, 3),
                o._flat_v[i][g].permute(0, 2, 1, 3),
            )

        def get_seq_length(self) -> int:
            return 0

        def get_max_length(self) -> int:
            return self.owner._gidx.shape[1]

        def get_mask_sizes(self, query_length: int):
            return self.owner._P, 0

        def lazy_initialization(self, *a, **k):  # always initialised
            pass

    return PagedGraphLayer


class GraphedPagedDecoder:
    """Decode one token for each of B sequences against the paged pool, as a graph."""

    def __init__(self, model, pool, max_batch: int, max_model_len: int, ctx_bucket: int = 32):
        from transformers.cache_utils import Cache

        impl = getattr(model.config, "_attn_implementation", None)
        if impl != "sdpa":
            raise ValueError(f"graphed paged decode needs attn_implementation='sdpa', got {impl!r}")
        self.model, self.pool = model, pool
        self.device = pool.spec.device
        if not str(self.device).startswith("cuda"):
            raise ValueError("graphed paged decode needs a CUDA device")
        self.bs = pool.block_size
        self.ctx_bucket = max(ctx_bucket, self.bs)
        self.max_batch = max(1, max_batch)
        # Wide enough for the *bucketed* gather of the longest reachable context,
        # so a rounded-up P never indexes past the table.
        width = ((max_model_len + self.bs + self.ctx_bucket - 1) // self.ctx_bucket) * self.ctx_bucket
        self.max_blocks = width // self.bs
        n, h, d = pool.spec.num_layers, pool.spec.num_kv_heads, pool.spec.head_dim
        self._flat_k = [t for t in pool.key_pool.view(n, -1, h, d)]
        self._flat_v = [t for t in pool.value_pool.view(n, -1, h, d)]
        mb, dev = self.max_batch, self.device
        self._tok = torch.zeros((mb, 1), dtype=torch.long, device=dev)
        self._pos = torch.zeros(mb, dtype=torch.long, device=dev)
        self._table = torch.zeros((mb, self.max_blocks), dtype=torch.long, device=dev)
        self._widx = torch.zeros(mb, dtype=torch.long, device=dev)
        # Gather index for every row, sized once for the longest reachable
        # context: reallocating would move an address a captured graph reads.
        self._gidx = torch.zeros((mb, self.max_blocks * self.bs), dtype=torch.long, device=dev)
        self._cp0 = torch.zeros(1, dtype=torch.long, device=dev)
        layer_cls = _make_layer_class()
        self.cache = Cache(layers=[layer_cls(self, i) for i in range(n)])
        self._mem_pool = torch.cuda.graph_pool_handle()
        self._graphs: Dict[Tuple[int, int], torch.cuda.CUDAGraph] = {}
        self._outs: Dict[Tuple[int, int], torch.Tensor] = {}
        self._b, self._P = 1, self.ctx_bucket
        self.replays = 0
        self.captures = 0

    @staticmethod
    def _batch_bucket(b: int) -> int:
        p = 1
        while p < b:
            p *= 2
        return p

    def _ctx_round(self, n: int) -> int:
        c = self.ctx_bucket
        return ((n + c - 1) // c) * c

    @torch.no_grad()
    def _run(self, b: int, P: int):
        """One decode step for rows [0:b) over a P-token gathered context."""
        self._b, self._P = b, P
        bs = self.bs
        pos, tbl = self._pos[:b], self._table[:b]
        rows = torch.arange(b, device=self.device)
        self._widx[:b].copy_(tbl[rows, pos // bs] * bs + pos % bs)
        ar = torch.arange(P, device=self.device)
        self._gidx[:b, :P].copy_(tbl[:, ar // bs] * bs + (ar % bs)[None, :])
        mask = (ar[None, :] <= pos[:, None])[:, None, None, :]
        return self.model(
            input_ids=self._tok[:b], position_ids=pos[:, None], past_key_values=self.cache,
            attention_mask=mask, cache_position=self._cp0, use_cache=True,
        ).logits[:, -1]

    @torch.no_grad()
    def _capture(self, b: int, P: int) -> None:
        # Warmup and capture run on the inputs the caller just set. They are
        # valid -- a duplicate write of identical values -- so no step ever
        # invents a write location.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._run(b, P)
        torch.cuda.current_stream().wait_stream(stream)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self._mem_pool):
            out = self._run(b, P)
        self._graphs[(b, P)] = g
        self._outs[(b, P)] = out
        self.captures += 1

    @torch.no_grad()
    def step(self, tokens: Sequence[int], ctx_lens: Sequence[int],
             block_tables: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Decode one token per row. `ctx_lens[i]` is the number of tokens already
        cached for row i; the new token is written at that logical position
        (its block must already be allocated). Returns fresh logits `(n, vocab)`.
        """
        n = len(tokens)
        if n == 0:
            raise ValueError("step() needs at least one sequence")
        b = self._batch_bucket(n)
        if b > self.max_batch:
            raise ValueError(f"batch {n} exceeds max_batch {self.max_batch}")
        P = self._ctx_round(max(ctx_lens) + 1)
        if P > self._gidx.shape[1]:
            raise ValueError(f"context {max(ctx_lens) + 1} exceeds the table width")

        toks = list(tokens) + [tokens[0]] * (b - n)
        poss = list(ctx_lens) + [ctx_lens[0]] * (b - n)
        tabs = list(block_tables) + [block_tables[0]] * (b - n)
        self._tok[:b, 0].copy_(torch.tensor(toks, dtype=torch.long, device=self.device))
        self._pos[:b].copy_(torch.tensor(poss, dtype=torch.long, device=self.device))
        self._table[:b].zero_()
        for i, t in enumerate(tabs):
            t = t[: self.max_blocks]  # extra trailing blocks are never addressed
            self._table[i, : t.numel()] = t
        if (b, P) not in self._graphs:
            self._capture(b, P)
        self._graphs[(b, P)].replay()
        self.replays += 1
        return self._outs[(b, P)][:n].clone()
