"""Token sampling utilities shared by the continuous and speculative runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SamplingParams:
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    seed: Optional[int] = None
    stop_on_eos: bool = True

    def clamped(self, hard_max: int) -> "SamplingParams":
        return SamplingParams(
            max_new_tokens=max(1, min(self.max_new_tokens, hard_max)),
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
            stop_on_eos=self.stop_on_eos,
        )


def apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus filter over a (..., vocab) probability tensor."""
    if top_p >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # Keep the smallest prefix whose mass exceeds top_p (always keep the top-1).
    drop = cumulative - sorted_probs > top_p
    sorted_probs = sorted_probs.masked_fill(drop, 0.0)
    filtered = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
    total = filtered.sum(dim=-1, keepdim=True)
    return torch.where(total > 0, filtered / total, probs)


def logits_to_probs(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Convert logits to the sampling distribution actually used to draw a token."""
    logits = logits.float()
    if temperature <= 0.0:
        # Greedy expressed as a distribution, so the speculative sampler can still
        # reason about it as P(x)/Q(x).
        probs = torch.zeros_like(logits)
        probs.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
        return probs
    probs = torch.softmax(logits / temperature, dim=-1)
    return apply_top_p(probs, top_p)


def sample_from_probs(probs: torch.Tensor, generator: Optional[torch.Generator] = None) -> int:
    """Draw one token id from a 1-D probability vector."""
    probs = torch.clamp(probs, min=0.0)
    total = probs.sum()
    if not torch.isfinite(total) or total <= 0:
        probs = torch.ones_like(probs) / probs.numel()
    else:
        probs = probs / total
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def sample_logits(
    logits: torch.Tensor,
    params: SamplingParams,
    generator: Optional[torch.Generator] = None,
) -> int:
    probs = logits_to_probs(logits, params.temperature, params.top_p)
    if params.temperature <= 0.0:
        return int(probs.argmax().item())
    return sample_from_probs(probs, generator)
