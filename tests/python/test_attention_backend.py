"""
Tests for the block-wise attention backend wired into a real model (tier 3).

`test_block_attention.py` proves the kernel. This proves the *integration*,
which is where the bugs actually were: three separate defects, every one of
which produced fluent, plausible, wrong output rather than an error.

  1. Binding a context length captured *before* the forward pass, so the query
     could not attend to its own key — 0.72 max logit deviation while every
     layer reported taking the fast path.
  2. Returning `(batch, heads, q_len, dim)` where the caller reshapes on
     `(batch, q_len, heads, dim)`, interleaving the head and position axes.
  3. Treating a `None` attention mask as "attend to everything". Transformers'
     `create_causal_mask()` returns `None` when the mask would be purely causal
     and expects the backend to enforce causality itself, as sdpa and flash do —
     so prefill ran bidirectional attention.

Each has a test below. None of them would have been caught by checking for
exceptions, which is why these compare against eager attention instead.
"""

from __future__ import annotations

import pytest
import torch

from memforge_ml.kernels import attention_backend as ab
from memforge_ml.runtime.continuous_engine import ContinuousBatchingEngine
from memforge_ml.runtime.kv_pool import ModelKVSpec, PagedKVPool
from memforge_ml.runtime.paged_cache import PagedCache
from memforge_ml.runtime.sampling import SamplingParams

BLOCK = 4
EPS = 2e-5


@pytest.fixture(autouse=True)
def restore_attn_implementation(tiny_model):
    """
    The backend is installed by mutating the model config, and the model fixture
    is session-scoped, so a test that enables it would leak into every later one.
    """
    original = getattr(tiny_model.config, "_attn_implementation", "eager")
    yield
    tiny_model.config._attn_implementation = original


def decode_run(model, spec, lazy, ctx=21, steps=6):
    """Prefill then decode, returning the prefill logits, decode logits and stats."""
    num_blocks = (ctx + steps) // BLOCK + 3
    pool = PagedKVPool(spec, num_blocks=num_blocks, block_size=BLOCK)
    cache = PagedCache(pool, list(range(num_blocks)), lazy_read=lazy)
    if lazy:
        assert ab.enable_on(model)
    else:
        model.config._attn_implementation = "eager"

    prompt = [2 + (i * 11) % 200 for i in range(ctx)]
    with torch.no_grad():
        out = model(
            input_ids=torch.tensor([prompt], dtype=torch.long),
            past_key_values=cache, use_cache=True,
        )
    prefill = out.logits[0, -1].clone()

    logits, stats = [], []
    for _ in range(steps):
        token = int(out.logits[0, -1].argmax())
        logits.append(out.logits[0, -1].clone())
        cache.begin_step()
        ids = torch.tensor([[token]], dtype=torch.long)
        if lazy:
            with ab.paged_attention_context(pool, cache) as bound:
                with torch.no_grad():
                    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
                stats.append((bound.used, bound.fallbacks))
        else:
            with torch.no_grad():
                out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    return prefill, logits, stats


# ------------------------------------------------------------- registration

def test_register_is_idempotent_and_registers_a_mask_builder():
    """
    Registering the attention function alone is not enough.

    Transformers keeps a separate mask registry keyed on the same name, and an
    unrecognised name there means no mask is built.
    """
    assert ab.register() is True
    assert ab.register() is True  # idempotent

    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    assert ab.BACKEND_NAME in ALL_ATTENTION_FUNCTIONS
    assert ab.BACKEND_NAME in ALL_MASK_ATTENTION_FUNCTIONS


def test_enable_and_detect_on_a_model(tiny_model):
    assert ab.enable_on(tiny_model) is True
    assert ab.is_enabled_on(tiny_model) is True
    tiny_model.config._attn_implementation = "eager"
    assert ab.is_enabled_on(tiny_model) is False


# ----------------------------------------------------------- equivalence

def test_prefill_matches_eager_exactly(tiny_model):
    """
    Bug 3's regression test.

    Prefill falls back to the materialised path, so this isolates whether that
    fallback applies causality. When it did not, prefill was bidirectional and
    deviated by 0.16 — close enough to read as a numerical quirk.
    """
    spec = ModelKVSpec.from_config(tiny_model.config, torch.float32, "cpu")
    ref, _, _ = decode_run(tiny_model, spec, lazy=False)
    got, _, _ = decode_run(tiny_model, spec, lazy=True)
    assert torch.allclose(ref, got, atol=EPS), (
        f"prefill diverged by {float((ref - got).abs().max()):.3e}; "
        "the fallback is probably not applying a causal mask"
    )


def test_decode_matches_eager_and_every_layer_uses_the_kernel(tiny_model):
    """
    Bugs 1 and 2's regression test, plus the thing that makes the test
    meaningful: asserting the kernel was *taken*. A silent fallback would make
    this pass while measuring nothing.
    """
    spec = ModelKVSpec.from_config(tiny_model.config, torch.float32, "cpu")
    _, ref, _ = decode_run(tiny_model, spec, lazy=False)
    _, got, stats = decode_run(tiny_model, spec, lazy=True)

    assert stats, "no decode steps recorded"
    for step, (used, fallbacks) in enumerate(stats):
        assert used == spec.num_layers, f"step {step}: only {used} layers block-wise"
        assert fallbacks == 0, f"step {step}: {fallbacks} layers fell back"

    worst = max(float((a - b).abs().max()) for a, b in zip(ref, got))
    assert worst < EPS, f"decode diverged by {worst:.3e}"


# --------------------------------------------------------------- the binding

def test_binding_reads_the_context_length_after_the_write(tiny_model):
    """
    Bug 1 directly: the context length must include the key just written.

    The binding holds the cache rather than a snapshot precisely so this cannot
    drift; this pins the property.
    """
    spec = ModelKVSpec.from_config(tiny_model.config, torch.float32, "cpu")
    pool = PagedKVPool(spec, num_blocks=16, block_size=BLOCK)
    cache = PagedCache(pool, list(range(16)), lazy_read=True)

    with ab.paged_attention_context(pool, cache) as bound:
        assert bound.context_length == 0
        cache.layers[0].position = 7          # stands in for a write
        assert bound.context_length == 7, "binding cached a stale length"
        assert bound.block_table is cache.block_table_tensor_cached()


def test_bindings_nest_and_restore():
    """A speculative pass inside a decode step must not strand the outer binding."""
    assert ab._current() is None
    with ab.paged_attention_context("outer-pool", "outer-cache") as outer:
        assert ab._current() is outer
        with ab.paged_attention_context("inner-pool", "inner-cache") as inner:
            assert ab._current() is inner
        assert ab._current() is outer, "inner binding did not restore the outer one"
    assert ab._current() is None


def test_no_binding_falls_back_cleanly(tiny_model):
    """
    Attention outside a bound context must still compute correctly.

    Prefill does exactly this, so the fallback is a normal path rather than an
    error case.
    """
    spec = ModelKVSpec.from_config(tiny_model.config, torch.float32, "cpu")
    assert ab.enable_on(tiny_model)
    pool = PagedKVPool(spec, num_blocks=16, block_size=BLOCK)
    cache = PagedCache(pool, list(range(16)))

    assert ab._current() is None
    with torch.no_grad():
        out = tiny_model(
            input_ids=torch.tensor([[2, 3, 4, 5]], dtype=torch.long),
            past_key_values=cache, use_cache=True,
        )
    assert torch.isfinite(out.logits).all()


# ----------------------------------------------------------------- masking

def test_mask_permissiveness_check():
    """A padded batch must not take the block-wise path, which ignores the mask."""
    assert ab._mask_is_permissive(None) is True
    assert ab._mask_is_permissive(torch.zeros(1, 1, 1, 4)) is True
    assert ab._mask_is_permissive(torch.tensor([[[[0.0, -1e9]]]])) is False
    assert ab._mask_is_permissive(torch.ones(1, 1, 1, 4, dtype=torch.bool)) is True
    assert ab._mask_is_permissive(
        torch.tensor([[[[True, False]]]], dtype=torch.bool)
    ) is False


def test_implicit_causal_mask_matches_an_explicit_one():
    """
    Bug 3's unit-level test.

    With a right-aligned kv window, query i may see keys up to
    `kv_len - q_len + i`. An off-by-one here leaks one future token, which is
    exactly the kind of error that still produces readable output.
    """
    torch.manual_seed(0)
    q_len, kv_len = 4, 9
    scores = torch.randn(1, 2, q_len, kv_len)

    got = ab._apply_implicit_causal_mask(scores)
    allowed = torch.ones(q_len, kv_len, dtype=torch.bool).tril(diagonal=kv_len - q_len)
    assert torch.equal(got[..., allowed], scores[..., allowed])
    assert torch.all(got[..., ~allowed] == torch.finfo(scores.dtype).min)

    # A single query needs no mask at all: everything before it is visible.
    single = torch.randn(1, 2, 1, 5)
    assert torch.equal(ab._apply_implicit_causal_mask(single), single)


def test_eager_fallback_matches_transformers(tiny_model):
    """
    The fallback must be *exactly* transformers' eager attention, since prefill
    depends on it. Bug 2 lived here.
    """
    from transformers.models.llama.modeling_llama import eager_attention_forward

    class _Module(torch.nn.Module):
        def __init__(self, groups):
            super().__init__()
            self.num_key_value_groups = groups
            self.training = False

    torch.manual_seed(0)
    heads, kv_heads, q_len, kv_len, dim = 4, 2, 5, 9, 8
    module = _Module(heads // kv_heads)
    query = torch.randn(1, heads, q_len, dim)
    key = torch.randn(1, kv_heads, kv_len, dim)
    value = torch.randn(1, kv_heads, kv_len, dim)
    scaling = dim ** -0.5

    causal = torch.zeros(1, 1, q_len, kv_len).masked_fill(
        torch.ones(q_len, kv_len).tril(diagonal=kv_len - q_len) == 0, float("-inf")
    )
    for mask in (None, causal):
        ref, _ = eager_attention_forward(module, query, key, value, mask, scaling=scaling)
        got = ab._eager_attention(module, query, key, value, mask, scaling)
        assert got.shape == ref.shape
        if mask is not None:
            assert torch.equal(ref, got)


# -------------------------------------------------------- engine integration

def test_engine_block_attention_is_byte_identical(tiny_model, stub_tokenizer):
    """End to end through the engine, which is how it will actually be used."""
    prompts = [f"engine level check number {i} with some length" for i in range(3)]
    params = SamplingParams(max_new_tokens=5, temperature=0.0, stop_on_eos=False)

    def run(**kwargs):
        tiny_model.config._attn_implementation = "eager"
        engine = ContinuousBatchingEngine(
            tiny_model, stub_tokenizer, block_size=BLOCK, num_blocks=256,
            max_num_seqs=4, max_model_len=256, enable_prefix_sharing=False,
            batched_decode=False, **kwargs,
        )
        return [r["token_ids"] for r in engine.generate(prompts, params)], engine.telemetry()

    baseline, _ = run()
    blockwise, telemetry = run(block_attention=True)

    assert baseline == blockwise, "block attention changed the engine's output"
    assert telemetry["block_attention_enabled"] is True
    assert telemetry["blockwise_layer_calls"] > 0, "the kernel was never used"
    assert telemetry["blockwise_fallbacks"] == 0


def test_engine_refuses_mirror_and_block_attention_together(tiny_model, stub_tokenizer):
    """
    They solve the same problem in opposite ways, so enabling both is a
    configuration mistake worth refusing rather than silently resolving.
    """
    tiny_model.config._attn_implementation = "eager"
    with pytest.raises(ValueError, match="mutually exclusive"):
        ContinuousBatchingEngine(
            tiny_model, stub_tokenizer, block_size=BLOCK, num_blocks=64,
            max_num_seqs=2, max_model_len=128,
            decode_view_cache=True, block_attention=True,
        )


def test_lazy_read_rejects_a_mirror_capacity():
    spec = ModelKVSpec(
        num_layers=2, num_kv_heads=2, head_dim=8, dtype=torch.float32, device="cpu"
    )
    pool = PagedKVPool(spec, num_blocks=8, block_size=BLOCK)
    with pytest.raises(ValueError, match="mutually exclusive"):
        PagedCache(pool, [0, 1], lazy_read=True, view_capacity=16)
