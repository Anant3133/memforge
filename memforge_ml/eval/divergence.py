"""
Is a token mismatch a bug, or half-precision noise on a near-tie?

Byte-identical output between two greedy runs is a guarantee only when both
compute the same numbers. Batched decode, K+1-wide verification and
single-sequence decode are different reduction orders; in fp16/bf16 that moves
a logit by a few units in the last place, and if the model's top two tokens are
within that noise, argmax flips. Nothing is wrong with the algorithm.

That claim can be tested instead of asserted. At the *first* position where two
runs disagree the contexts are identical, so one clean single-sequence forward
pass gives the logits both runs were trying to argmax. If the two chosen tokens
are within a tolerance scaled to the dtype's rounding error, the mismatch is
explained. If a token is far below the top, it is a real bug and is reported
as one.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

_EPS = {torch.float16: 2.0 ** -10, torch.bfloat16: 2.0 ** -7, torch.float32: 2.0 ** -23}


def near_tie_tolerance(dtype: torch.dtype, logit_scale: float, layers: int = 32) -> float:
    """
    Logit gap below which two tokens are indistinguishable in this dtype.

    One rounding moves a value by ~eps*|x|; a forward pass accumulates it over
    the depth, but errors add as a random walk, so ~sqrt(layers). This is a
    deliberately generous bound; a real bug shows up as a gap of many units.
    """
    eps = _EPS.get(dtype, 2.0 ** -10)
    return 4.0 * eps * max(1.0, abs(logit_scale)) * (layers ** 0.5)


def first_divergence(a: Sequence[int], b: Sequence[int]) -> Optional[int]:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


@torch.no_grad()
def explain_divergence(
    model, prompt_ids: Sequence[int], a: Sequence[int], b: Sequence[int],
    device: str, tol: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    None if the runs are identical; otherwise a report for the first mismatch:
    the two tokens, their logit gap in a clean single-sequence pass, the
    tolerance, and `explained` (gap <= tol, i.e. a near-tie).
    """
    i = first_divergence(a, b)
    if i is None:
        return None
    if i >= len(a) or i >= len(b):
        return {"index": i, "kind": "length", "explained": False,
                "note": f"runs have different lengths ({len(a)} vs {len(b)})"}
    ctx = list(prompt_ids) + list(a[:i])
    ids = torch.tensor([ctx], dtype=torch.long, device=device)
    logits = model(input_ids=ids, use_cache=False).logits[0, -1].float()
    za, zb = float(logits[a[i]]), float(logits[b[i]])
    top = float(logits.max())
    dtype = next(model.parameters()).dtype
    layers = getattr(getattr(model, "config", None), "num_hidden_layers", 32)
    tol = near_tie_tolerance(dtype, top, layers) if tol is None else tol
    gap = abs(za - zb)
    return {
        "index": i, "kind": "token", "token_a": int(a[i]), "token_b": int(b[i]),
        "logit_gap": round(gap, 5), "tolerance": round(tol, 5),
        "gap_below_top": round(top - min(za, zb), 5),
        "explained": gap <= tol,
    }


def summarize(reports: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    real = [r for r in reports if r is not None]
    return {
        "sequences": len(reports),
        "identical": len(reports) - len(real),
        "diverged": len(real),
        "explained_near_ties": sum(1 for r in real if r["explained"]),
        "unexplained": sum(1 for r in real if not r["explained"]),
        "equivalent": all(r["explained"] for r in real),
    }
