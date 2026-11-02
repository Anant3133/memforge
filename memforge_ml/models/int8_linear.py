"""
Weight-only int8 linear layers, so a 14B model fits on a card that has ~25 GB free.

Per-output-channel symmetric quantisation: `W ~ Wq * scale[:, None]` with
`Wq` in int8. Forward is `(x @ Wq^T) * scale`, i.e. the scale is applied to the
*output*, which is exactly equal to scaling the weight and avoids materialising
a scaled copy. Only the int8 -> compute-dtype cast of one layer's weight is
temporary, so peak extra memory is one layer, not the model.

This is a memory device, not a speed one (the cast costs bandwidth), and it
changes numerics. Anything compared against an unquantised model is invalid;
only comparisons *within* the quantised model (paged vs contiguous, with vs
without prefix sharing) are meaningful. That is how the 14B runs use it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Int8Linear(nn.Module):
    def __init__(self, weight_q: torch.Tensor, scale: torch.Tensor, bias=None):
        super().__init__()
        self.register_buffer("weight_q", weight_q)  # (out, in) int8
        self.register_buffer("scale", scale)        # (out,) compute dtype
        self.bias = bias
        self.out_features, self.in_features = weight_q.shape

    @classmethod
    @torch.no_grad()
    def from_linear(cls, lin: nn.Linear, device) -> "Int8Linear":
        w = lin.weight.detach().to(device=device, dtype=torch.float32)
        scale = (w.abs().amax(dim=1) / 127.0).clamp_min(1e-8)
        wq = torch.round(w / scale[:, None]).clamp_(-127, 127).to(torch.int8)
        bias = None if lin.bias is None else nn.Parameter(
            lin.bias.detach().to(device=device, dtype=lin.weight.dtype), requires_grad=False)
        return cls(wq, scale.to(lin.weight.dtype), bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight_q.to(x.dtype)) * self.scale.to(x.dtype)
        return y if self.bias is None else y + self.bias


@torch.no_grad()
def quantize_linears_int8(model: nn.Module, device, skip=("lm_head",)) -> int:
    """Replace every nn.Linear (except names in `skip`) with an Int8Linear on `device`."""
    count = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and name not in skip:
                setattr(parent, name, Int8Linear.from_linear(child, device))
                count += 1
    return count
