"""
Tests for speculative decoding over the paged cache.

The load-bearing property is losslessness: speculation must not change what the
target model would have produced. At temperature 0 that is exactly checkable --
greedy speculative decoding has to equal greedy decoding with the target alone.
"""

import pytest
import torch

from memforge_ml.runtime.sampling import SamplingParams
from memforge_ml.runtime.speculative_runtime import (
    HAS_CPP_ENGINE,
    PagedSpeculativeEngine,
)

pytestmark = pytest.mark.skipif(
    not HAS_CPP_ENGINE, reason="native memforge_c extension is not built"
)


def make_engine(target, draft, tokenizer, **kwargs):
    options = dict(lookahead_k=4, block_size=4, num_blocks=64, max_model_len=128)
    options.update(kwargs)
    return PagedSpeculativeEngine(target, tokenizer, draft, tokenizer, **options)


def greedy_reference(model, prompt_ids, num_tokens, eos_token_id):
    """Plain greedy decoding with the target model alone."""
    ids = torch.tensor([prompt_ids], dtype=torch.long)
    out = []
    with torch.no_grad():
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
        logits = model(input_ids=ids, past_key_values=cache, use_cache=True).logits
        for _ in range(num_tokens):
            token = int(logits[0, -1].argmax())
            out.append(token)
            if token == eos_token_id:
                break
            logits = model(
                input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True
            ).logits
    return out


def test_greedy_speculative_equals_greedy_target(
    tiny_model, tiny_draft_model, stub_tokenizer, prompt_ids
):
    """The whole point of rejection sampling: identical output, fewer target passes."""
    num_tokens = 16
    reference = greedy_reference(
        tiny_model, prompt_ids, num_tokens, stub_tokenizer.eos_token_id
    )

    engine = make_engine(tiny_model, tiny_draft_model, stub_tokenizer)
    result = engine.generate(
        prompt_ids, SamplingParams(max_new_tokens=num_tokens, temperature=0.0, top_p=1.0)
    )

    assert result["token_ids"] == reference


def test_a_perfect_draft_accepts_every_proposal(tiny_model, stub_tokenizer, prompt_ids):
    """
    Using the target as its own draft is the best case: every proposal matches,
    so each target pass should confirm K draft tokens plus a bonus token, and the
    number of target forward passes should drop by roughly that factor.
    """
    k = 4
    engine = make_engine(tiny_model, tiny_model, stub_tokenizer, lookahead_k=k)
    result = engine.generate(
        prompt_ids, SamplingParams(max_new_tokens=20, temperature=0.0, top_p=1.0)
    )

    assert result["acceptance_rate"] == 1.0
    assert result["resampled_tokens"] == 0
    assert result["target_forward_passes"] < result["generated_tokens"]
    assert result["tokens_per_target_pass"] == pytest.approx(k + 1, abs=0.5)
    assert result["rolled_back_tokens"] == 0


def test_an_uncorrelated_draft_is_still_lossless(
    tiny_model, tiny_draft_model, stub_tokenizer, prompt_ids
):
    """
    Worst case: a draft that never agrees with the target. Speculation then buys
    nothing (one token per target pass), but the output must still be exactly
    what the target alone would have produced -- that is the guarantee.
    """
    num_tokens = 16
    reference = greedy_reference(
        tiny_model, prompt_ids, num_tokens, stub_tokenizer.eos_token_id
    )

    engine = make_engine(tiny_model, tiny_draft_model, stub_tokenizer)
    result = engine.generate(
        prompt_ids, SamplingParams(max_new_tokens=num_tokens, temperature=0.0, top_p=1.0)
    )

    assert result["token_ids"] == reference
    assert result["tokens_per_target_pass"] >= 1.0


def test_generates_exactly_the_requested_token_count(
    tiny_model, tiny_draft_model, stub_tokenizer, prompt_ids
):
    """The bonus token must never push the output past max_new_tokens."""
    engine = make_engine(tiny_model, tiny_draft_model, stub_tokenizer)
    for num_tokens in (1, 2, 5, 7, 12):
        result = engine.generate(
            prompt_ids,
            SamplingParams(max_new_tokens=num_tokens, temperature=0.0, top_p=1.0),
        )
        assert result["generated_tokens"] == num_tokens, num_tokens
        assert len(result["token_sources"]) == num_tokens


def test_every_token_is_attributed_to_a_known_source(
    tiny_model, tiny_draft_model, stub_tokenizer, prompt_ids
):
    engine = make_engine(tiny_model, tiny_draft_model, stub_tokenizer)
    result = engine.generate(
        prompt_ids, SamplingParams(max_new_tokens=20, temperature=0.8, top_p=0.9, seed=7)
    )

    assert set(result["token_sources"]) <= {"draft", "resample", "bonus"}
    assert len(result["token_sources"]) == result["generated_tokens"]


def test_acceptance_accounting_is_self_consistent(
    tiny_model, tiny_draft_model, stub_tokenizer, prompt_ids
):
    engine = make_engine(tiny_model, tiny_draft_model, stub_tokenizer)
    result = engine.generate(
        prompt_ids, SamplingParams(max_new_tokens=20, temperature=0.0, top_p=1.0)
    )

    accepted = result["token_sources"].count("draft")
    corrected = result["token_sources"].count("resample")
    bonus = result["token_sources"].count("bonus")

    assert accepted + corrected + bonus == result["generated_tokens"]
    assert result["accepted_tokens"] >= accepted  # extra accepts may be trimmed at the limit
    assert 0.0 <= result["acceptance_rate"] <= 1.0
    assert result["drafted_tokens"] >= result["accepted_tokens"]


def test_blocks_are_released_when_generation_ends(
    tiny_model, tiny_draft_model, stub_tokenizer, prompt_ids
):
    engine = make_engine(tiny_model, tiny_draft_model, stub_tokenizer)
    total = engine.allocator.get_total_blocks()

    engine.generate(prompt_ids, SamplingParams(max_new_tokens=16, temperature=0.0))

    assert engine.allocator.get_num_free_blocks() == total


def test_blocks_are_released_even_if_the_stream_is_abandoned(
    tiny_model, tiny_draft_model, stub_tokenizer, prompt_ids
):
    """A client disconnecting mid-stream must not leak physical blocks."""
    engine = make_engine(tiny_model, tiny_draft_model, stub_tokenizer)
    total = engine.allocator.get_total_blocks()

    stream = engine.stream(prompt_ids, SamplingParams(max_new_tokens=32, temperature=0.0))
    next(stream)
    next(stream)
    stream.close()  # simulates the caller going away

    assert engine.allocator.get_num_free_blocks() == total


def test_stream_yields_one_event_per_token(
    tiny_model, tiny_draft_model, stub_tokenizer, prompt_ids
):
    engine = make_engine(tiny_model, tiny_draft_model, stub_tokenizer)
    generator = engine.stream(
        prompt_ids, SamplingParams(max_new_tokens=10, temperature=0.0, top_p=1.0)
    )

    events = []
    while True:
        try:
            events.append(next(generator))
        except StopIteration as stop:
            summary = stop.value
            break

    assert len(events) == 10
    assert [e.index for e in events] == list(range(10))
    assert [e.token_id for e in events] == summary["token_ids"]


def test_rollback_is_recorded_when_proposals_are_rejected(
    tiny_model, tiny_draft_model, stub_tokenizer, prompt_ids
):
    """Rejections should show up as rolled-back cache positions."""
    engine = make_engine(tiny_model, tiny_draft_model, stub_tokenizer)
    result = engine.generate(
        prompt_ids, SamplingParams(max_new_tokens=24, temperature=0.0, top_p=1.0)
    )

    if result["resampled_tokens"] > 0:
        assert result["rolled_back_tokens"] > 0


def test_mismatched_vocabularies_are_rejected(tiny_model, stub_tokenizer):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=128,  # different from the tiny model's 256
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    mismatched = LlamaForCausalLM(config).eval()

    with pytest.raises(ValueError, match="shared vocabulary"):
        make_engine(tiny_model, mismatched, stub_tokenizer)


def test_pool_exhaustion_raises_rather_than_corrupting(
    tiny_model, tiny_draft_model, stub_tokenizer, prompt_ids
):
    """A KV pool too small for the request must fail loudly."""
    engine = make_engine(
        tiny_model, tiny_draft_model, stub_tokenizer, block_size=4, num_blocks=3
    )
    with pytest.raises(MemoryError, match="KV pool exhausted"):
        engine.generate(prompt_ids, SamplingParams(max_new_tokens=64, temperature=0.0))

    # And the failed attempt must still have returned its blocks.
    assert engine.allocator.get_num_free_blocks() == engine.allocator.get_total_blocks()
