"""
Low-level read/write kernels for the paged KV cache (Phase A).

Everything here is a pure function over a `PagedKVPool` and index tensors, with
no engine state, so each one can be benchmarked and differentially tested in
isolation. That separation is the point: the gather is on the hot path of every
decode step for every layer, so it needs to be replaceable — by a Triton kernel
on GPU, by a different layout on CPU — without touching the cache or the engine.

`PagedKVPool.gather`/`write` remain the reference implementations. Every kernel
here is checked against them for byte-equality, because a wrong KV read does not
crash: it produces fluent, confident, wrong text.
"""

from memforge_ml.kernels.batched_decode import BatchedDecodeBuffers, assemble_batch
from memforge_ml.kernels.block_attention import (
    paged_attention_blockwise,
    reference_attention,
)
from memforge_ml.kernels.paged_gather import (
    flat_slot_index,
    gather_kv,
    scatter_kv,
)

__all__ = [
    # tier 1a -- gather/scatter over a flattened pool view
    "flat_slot_index",
    "gather_kv",
    "scatter_kv",
    # tier 1b -- batched decode assembly without pad+cat
    "BatchedDecodeBuffers",
    "assemble_batch",
    # tier 3 -- block-wise attention, never materialising the context
    "paged_attention_blockwise",
    "reference_attention",
]
