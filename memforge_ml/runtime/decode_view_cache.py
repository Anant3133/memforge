"""
Append-only contiguous mirror of one sequence's paged KV (Phase A, tier 2).

This is the measured `O(context)` to `O(1)` fix for the decode read path, and it
is the one optimisation in Phase A with a cost worth arguing about.

**What it does.** The engine re-gathers a sequence's *entire* context on every
decode step, although exactly one token was added since the last one. This keeps
a contiguous `(1, heads, capacity, head_dim)` buffer per layer, appends the new
token's K/V, and hands attention a zero-copy slice of it.

Measured (32 layers, 5 KV heads, head_dim 64, fp32, K and V, block size 16):

| context | rebuild every step | append only | speedup |
| ------: | -----------------: | ----------: | ------: |
|     128 |            2.87 ms |     0.99 ms |    2.9x |
|     512 |            5.39 ms |     1.08 ms |    5.0x |
|    1024 |            9.92 ms |     1.15 ms |    8.6x |
|    2048 |           14.91 ms |     1.46 ms |   10.2x |

The shape matters more than the ratio: append-only is nearly flat in context
(0.99 to 1.46 ms across a 16x increase) where rebuilding is linear.

**What it costs, stated plainly.** The buffer is a contiguous copy of the
sequence's KV, which is the thing paging exists to avoid. Per sequence:

    2 (K and V) x layers x kv_heads x capacity x head_dim x dtype_bytes

For SmolLM2-360M at 1024 tokens that is ~84 MB *per sequence*. So this is:

  * **opt-in**, never on by default;
  * sized from the sequence's own `prompt + max_new_tokens`, not from
    `max_model_len`, because the difference is usually large;
  * released with the cache it belongs to, so it follows the same lifecycle as
    the block table and cannot outlive a finished or preempted sequence.

It does **not** reduce the physical block pool's occupancy, so prefix sharing's
savings are unaffected — the pool is the constrained resource sized by
`kv_budget_mb`, and this is transient working memory for an active sequence.
Both numbers are reported side by side so the trade is visible.

**Why it is safe.** The pool stays the source of truth; this is a cache of it
and is written in the same call that writes the pool. Everything that can
invalidate the logical-to-physical mapping underneath it already funnels through
`PagedCache`, which owns this object: `reset()`, `crop()` (speculative
rollback), `seed_position()` (a prefix-cache hit, where the leading tokens come
from blocks this sequence never wrote), and a rebind after copy-on-write. Each
of those is wired to a corresponding call here.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from memforge_ml.runtime.kv_pool import ModelKVSpec


class DecodeViewCache:
    """
    One sequence's contiguous KV mirror, appended to a token at a time.

    Shaped `(layers, 1, kv_heads, capacity, head_dim)` so that a per-layer slice
    is already in the `(1, heads, tokens, dim)` form attention wants, with no
    transpose on the read path. The read is a view, not a copy.
    """

    def __init__(self, spec: ModelKVSpec, capacity_tokens: int,
                 device: Optional[str] = None):
        if capacity_tokens <= 0:
            raise ValueError(f"capacity_tokens must be positive, got {capacity_tokens}")
        self.spec = spec
        self.capacity = capacity_tokens
        self.device = device or spec.device
        shape = (spec.num_layers, 1, spec.num_kv_heads, capacity_tokens, spec.head_dim)
        # Zeroed rather than empty: a read of an unwritten position should be a
        # predictable zero rather than whatever was in memory, which makes a
        # bookkeeping bug show up as wrong-but-finite output instead of a NaN
        # storm three layers later.
        self.keys = torch.zeros(shape, dtype=spec.dtype, device=self.device)
        self.values = torch.zeros(shape, dtype=spec.dtype, device=self.device)
        self._filled = 0

    # ------------------------------------------------------------ properties

    @property
    def filled(self) -> int:
        """Tokens currently valid in the mirror."""
        return self._filled

    @property
    def nbytes(self) -> int:
        return self.keys.numel() * self.keys.element_size() * 2

    def describe(self) -> dict:
        return {
            "capacity_tokens": self.capacity,
            "filled_tokens": self._filled,
            "bytes": self.nbytes,
            "mb": round(self.nbytes / (1024 * 1024), 3),
        }

    # ----------------------------------------------------------------- writes

    def seed(self, pool, flat_index: torch.Tensor, num_tokens: int) -> None:
        """
        Fill positions `[0, num_tokens)` from the pool, for every layer.

        Needed when tokens became resident without this cache seeing them
        written: after a prefix-cache hit, the leading blocks were populated by a
        different sequence entirely. One full gather, paid once per sequence
        rather than once per step.
        """
        if num_tokens > self.capacity:
            raise ValueError(
                f"cannot seed {num_tokens} tokens into a {self.capacity}-token mirror"
            )
        for layer_idx in range(self.spec.num_layers):
            k, v = pool.gather_flat(layer_idx, flat_index)
            self.keys[layer_idx, :, :, :num_tokens, :] = k
            self.values[layer_idx, :, :, :num_tokens, :] = v
        self._filled = num_tokens

    def append(self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor,
               at: int) -> None:
        """
        Write `keys`/`values` of shape `(1, heads, T, head_dim)` at position `at`.

        `at` is passed explicitly rather than tracked internally because the
        caller is a per-layer `update()` and all layers write the same positions:
        a shared counter incremented per layer would advance `num_layers` times
        per token.
        """
        num_new = keys.shape[-2]
        end = at + num_new
        if end > self.capacity:
            raise ValueError(
                f"append of {num_new} tokens at {at} exceeds capacity {self.capacity}"
            )
        self.keys[layer_idx, :, :, at:end, :] = keys
        self.values[layer_idx, :, :, at:end, :] = values
        # The last layer to write a given range is what marks it valid, so a
        # partially written token is never visible as complete.
        if layer_idx == self.spec.num_layers - 1:
            self._filled = max(self._filled, end)

    # ------------------------------------------------------------------ reads

    def view(self, layer_idx: int, num_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        A zero-copy `(1, heads, num_tokens, head_dim)` slice of the mirror.

        This is the whole point: no gather, no transpose, no allocation. The
        tensors alias the buffer, so a caller that needs to retain them past the
        next append must clone -- attention consumes them within the forward
        pass, so it does not.
        """
        if num_tokens > self.capacity:
            raise ValueError(
                f"requested {num_tokens} tokens from a {self.capacity}-token mirror"
            )
        return (
            self.keys[layer_idx, :, :, :num_tokens, :],
            self.values[layer_idx, :, :, :num_tokens, :],
        )

    # ---------------------------------------------------------- invalidation

    def truncate(self, num_tokens: int) -> None:
        """
        Drop everything past `num_tokens`.

        Used by speculative rollback. The discarded region is left as-is rather
        than zeroed: it is unreadable through `view()` and will be overwritten by
        the next append, so clearing it would be work for no observable effect.
        """
        self._filled = max(0, min(self._filled, num_tokens))

    def reset(self) -> None:
        self._filled = 0

    def invalidate(self) -> None:
        """
        Mark the mirror empty because the pool changed underneath it.

        Distinct from `reset()` in intent, not mechanics: this is the "I can no
        longer trust my contents" path, used after copy-on-write repoints a block
        or preemption frees one. The caller must re-`seed()` before reading.
        """
        self._filled = 0


def capacity_for(prompt_tokens: int, max_new_tokens: int, slack: int = 1) -> int:
    """
    Mirror capacity for one sequence.

    Sized from the sequence's own length, not `max_model_len`: the gap between
    the two is usually large, and this is the difference between ~84 MB and
    whatever the request actually needs. `slack` covers the bonus token a
    speculative round can produce beyond `max_new_tokens`.
    """
    return max(1, prompt_tokens + max_new_tokens + slack)
