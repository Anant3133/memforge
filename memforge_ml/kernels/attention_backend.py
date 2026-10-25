"""
Wiring block-wise paged attention into a Hugging Face model (Phase A, tier 3).

`block_attention.py` has the kernel and the equivalence proof; this is what
makes a real model use it. Without this the kernel is a proven function nothing
calls, which is not an optimisation.

**The hook.** `LlamaAttention.forward` resolves its attention implementation
through `ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation, ...)`,
so registering an entry and setting `_attn_implementation` is enough — no
monkey-patching of model internals, and it works for every architecture that
goes through that registry.

**Why a thread-local binding.** The registered function receives
`(module, query, key, value, attention_mask, **kwargs)` and no reference to the
cache, so it cannot reach the pool or the block table on its own. The engine
binds them for the duration of a forward pass via `paged_attention_context()`.
Thread-local rather than a plain global because the server drives `engine.step()`
in a worker thread while other threads may run the speculative engine;
`contextvars` would be wrong here, since `loop.run_in_executor` does not
propagate context into the executor thread.

**Why it only takes over decode.** A single-query step needs no mask — every
prior token is visible — so block-wise attention over the pool is exactly
equivalent. Prefill is a multi-query causal problem with a real mask, so this
falls back to the materialised path there rather than reimplementing masking.
That is also where the win is: prefill reads the context once by construction,
while decode re-read all of it on every step.

**The memory claim is only true in lazy mode.** `PagedCache.update()` normally
returns the whole materialised context, so simply computing attention
differently would save nothing. With `lazy_read` enabled the cache returns just
the newly written tokens and this backend reads the rest from the pool block by
block — which is what makes working memory `O(block_size)` instead of
`O(context)`.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Optional, Tuple

import torch

from memforge_ml.kernels.block_attention import paged_attention_blockwise

#: Name this backend registers under, and the value to assign to
#: `model.config._attn_implementation`.
BACKEND_NAME = "memforge_paged"

_state = threading.local()


class _Binding:
    """
    What the attention function needs that its signature does not give it.

    Holds the *cache*, not a snapshot of its block table and length. That is
    deliberate: the context length must be the one *after* this step's key was
    written, because a query attends to its own key. Binding a number captured
    before the forward pass is wrong by exactly one token, and the result is
    close enough to look plausible rather than obviously broken — measured at
    0.72 max logit deviation while every layer reported success. Reading
    `cache.position` at attention time is correct by construction, since
    `update()` has already run for this layer by then.
    """

    __slots__ = ("pool", "cache", "used", "fallbacks")

    def __init__(self, pool, cache):
        self.pool = pool
        self.cache = cache
        self.used = 0
        self.fallbacks = 0

    @property
    def block_table(self) -> torch.Tensor:
        return self.cache.block_table_tensor_cached()

    @property
    def context_length(self) -> int:
        return self.cache.position


def _current() -> Optional[_Binding]:
    return getattr(_state, "binding", None)


@contextmanager
def paged_attention_context(pool, cache):
    """
    Bind the pool and one sequence's cache for the enclosed forward pass.

    Nests safely: the previous binding is restored on exit, so a speculative
    verification pass inside a decode step cannot strand the outer one.

    Yields the binding, whose `used` / `fallbacks` counters say how many layers
    actually took the block-wise path — worth asserting in tests, because a
    silent fallback looks exactly like success while measuring nothing.
    """
    previous = _current()
    binding = _Binding(pool, cache)
    _state.binding = binding
    try:
        yield binding
    finally:
        _state.binding = previous


def _mask_is_permissive(attention_mask: Optional[torch.Tensor]) -> bool:
    """
    True when the mask hides nothing, so ignoring it is safe.

    Block-wise attention over a single query attends to the whole context, which
    is correct only if nothing is masked out. For an unpadded single-sequence
    decode that holds; a padded batch would need the mask honoured, so that case
    falls back rather than quietly producing attention over padding.
    """
    if attention_mask is None:
        return True
    if attention_mask.dtype == torch.bool:
        return bool(attention_mask.all())
    # Additive float mask: 0 means visible, large negative means hidden.
    return bool(torch.all(attention_mask == 0))


def paged_attention_forward(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    **kwargs: Any,
) -> Tuple[torch.Tensor, None]:
    """
    Attention implementation registered as `memforge_paged`.

    Takes the block-wise path only when every precondition holds, and otherwise
    computes ordinary attention over whatever `key`/`value` it was handed. The
    fallback is not a failure mode: prefill and padded batches are expected to
    use it.
    """
    binding = _current()
    single_query = query.shape[0] == 1 and query.shape[2] == 1

    if (
        binding is not None
        and single_query
        and _mask_is_permissive(attention_mask)
        and binding.context_length > 0
    ):
        binding.used += 1
        out = paged_attention_blockwise(
            query,
            binding.pool.key_pool,
            binding.pool.value_pool,
            module.layer_idx,
            binding.block_table,
            binding.context_length,
            binding.pool.block_size,
            scale=scaling,
        )
        return _to_attention_output_layout(out), None

    if binding is not None:
        binding.fallbacks += 1
    return _eager_attention(module, query, key, value, attention_mask, scaling), None


def _to_attention_output_layout(attn_output: torch.Tensor) -> torch.Tensor:
    """
    Convert `(batch, heads, q_len, head_dim)` to the layout the caller expects.

    Transformers' own `eager_attention_forward` ends with
    `attn_output.transpose(1, 2).contiguous()`, and the attention module then
    does `attn_output.reshape(*input_shape, -1)` -- so the contract is
    `(batch, q_len, heads, head_dim)`, not the head-major form attention
    computes in. Returning the untransposed tensor interleaves the head and
    position axes during that reshape.

    Worth stating because of how it failed: not a crash, not a NaN, but a 0.70
    max logit deviation while every layer reported taking the fast path. Output
    that is wrong and plausible is the characteristic failure of this whole
    subsystem, which is why its tests compare against a reference rather than
    checking for exceptions.
    """
    return attn_output.transpose(1, 2).contiguous()


def _eager_attention(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float],
) -> torch.Tensor:
    """
    Plain scaled dot-product attention, with grouped-query expansion.

    Written out rather than delegated to transformers' own eager path so this
    module does not depend on a private helper's location, which has moved
    between releases.
    """
    import math

    num_heads, head_dim = query.shape[1], query.shape[3]
    num_kv_heads = key.shape[1]
    scaling = scaling if scaling is not None else 1.0 / math.sqrt(head_dim)

    k, v = key, value
    group = num_heads // num_kv_heads
    if group > 1:
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)

    scores = torch.matmul(query, k.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : k.shape[-2]]
    else:
        scores = _apply_implicit_causal_mask(scores)
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return _to_attention_output_layout(torch.matmul(weights, v))


def _apply_implicit_causal_mask(scores: torch.Tensor) -> torch.Tensor:
    """
    Apply causality when transformers hands us no mask at all.

    `create_causal_mask()` has an `allow_is_causal_skip` fast path: when the
    mask would be purely causal with no padding, it returns **None** and expects
    the attention implementation to enforce causality itself, the way sdpa and
    flash do via `is_causal=True`. A backend that treats a missing mask as
    "attend to everything" therefore runs *bidirectional* attention during
    prefill.

    That is how this failed, and it is worth recording: no crash, no NaN, just a
    0.16 max logit deviation on prefill that compounded to 0.71 through decode
    and read like a numerical quirk. Registering a mask builder for this backend
    did not help, because the skip happens before the registry is consulted.

    A single query position needs nothing: everything before it is visible.
    """
    q_len, kv_len = scores.shape[-2], scores.shape[-1]
    if q_len <= 1:
        return scores
    # Right-aligned: query i corresponds to absolute position
    # kv_len - q_len + i, so it may attend to keys up to that index.
    causal = torch.ones(q_len, kv_len, dtype=torch.bool, device=scores.device).tril(
        diagonal=kv_len - q_len
    )
    return scores.masked_fill(~causal, torch.finfo(scores.dtype).min)


def register() -> bool:
    """
    Add this backend to the model's attention registry.

    Idempotent, and returns False rather than raising when the installed
    transformers has no such registry: callers should then leave
    `_attn_implementation` alone and keep the materialised path, which is slower
    but correct. A hard failure here would make the engine unusable on a version
    that is otherwise perfectly fine.
    """
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except ImportError:
        return False

    try:
        if BACKEND_NAME not in ALL_ATTENTION_FUNCTIONS:
            ALL_ATTENTION_FUNCTIONS[BACKEND_NAME] = paged_attention_forward
    except Exception:
        return False

    # Registering the attention function is only half of it. Transformers keeps
    # a *separate* registry for mask construction, also keyed on
    # `_attn_implementation`, and an unrecognised name there means no mask is
    # built at all -- it passes None. Prefill then runs full bidirectional
    # attention instead of causal, which does not crash and does not produce
    # NaNs; it produced a 0.16 max logit deviation that looked like a numerical
    # quirk until the mask was actually inspected. Aliasing eager's builder
    # gives this backend exactly the mask eager would have received.
    try:
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

        if BACKEND_NAME not in ALL_MASK_ATTENTION_FUNCTIONS:
            ALL_MASK_ATTENTION_FUNCTIONS[BACKEND_NAME] = (
                ALL_MASK_ATTENTION_FUNCTIONS["eager"]
            )
    except Exception:
        # Without a mask builder the backend would silently lose causal masking,
        # which is far worse than not being available at all.
        return False

    return True


def enable_on(model) -> bool:
    """
    Point a loaded model at this backend.

    Returns False if registration failed, leaving the model untouched.
    """
    if not register():
        return False
    model.config._attn_implementation = BACKEND_NAME
    # Sub-configs exist on multimodal models and are what the attention layers
    # actually read on some architectures.
    for attr in ("text_config", "decoder"):
        sub = getattr(model.config, attr, None)
        if sub is not None and hasattr(sub, "_attn_implementation"):
            sub._attn_implementation = BACKEND_NAME
    return True


def is_enabled_on(model) -> bool:
    return getattr(model.config, "_attn_implementation", None) == BACKEND_NAME
