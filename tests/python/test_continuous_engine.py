"""
Tests for the continuous batching engine.

These cover the properties that make continuous batching worth having: requests
joining mid-flight, memory being reclaimed, sequences surviving preemption, and
the batched decode path giving the same answers as the sequential one.
"""

import pytest
import torch

from memforge_ml.runtime.continuous_engine import HAS_CPP_ENGINE, ContinuousBatchingEngine
from memforge_ml.runtime.sampling import SamplingParams

pytestmark = pytest.mark.skipif(
    not HAS_CPP_ENGINE, reason="native memforge_c extension is not built"
)

GREEDY = SamplingParams(max_new_tokens=8, temperature=0.0, top_p=1.0)


def make_engine(model, tokenizer, **kwargs):
    options = dict(
        block_size=4, num_blocks=64, max_num_seqs=4, max_model_len=128
    )
    options.update(kwargs)
    return ContinuousBatchingEngine(model, tokenizer, **options)


def test_generates_the_requested_number_of_tokens(tiny_model, stub_tokenizer):
    engine = make_engine(tiny_model, stub_tokenizer)
    result = engine.generate("hello world", GREEDY)[0]

    assert result["generated_tokens"] == 8
    assert len(result["token_ids"]) == 8
    assert result["finish_reason"] == "length"
    assert result["time_to_first_token_ms"] is not None


def test_batched_decode_matches_sequential_decode(tiny_model, stub_tokenizer):
    """The left-padded batched gather/scatter must not change any output."""
    prompts = ["alpha prompt", "a much longer beta prompt here", "gamma"]

    batched = make_engine(tiny_model, stub_tokenizer, batched_decode=True)
    sequential = make_engine(tiny_model, stub_tokenizer, batched_decode=False)

    batched_results = batched.generate(prompts, GREEDY)
    sequential_results = sequential.generate(prompts, GREEDY)

    for a, b in zip(batched_results, sequential_results):
        assert a["token_ids"] == b["token_ids"]

    # And the batched path really did use fewer forward passes.
    assert batched.total_forward_passes < sequential.total_forward_passes


def test_batching_needs_fewer_forward_passes_than_tokens(tiny_model, stub_tokenizer):
    engine = make_engine(tiny_model, stub_tokenizer, batched_decode=True)
    engine.generate(["one", "two", "three", "four"], GREEDY)

    telemetry = engine.telemetry()
    assert telemetry["tokens_generated"] == 32
    assert telemetry["tokens_per_forward_pass"] > 1.0


def test_blocks_are_fully_reclaimed_after_requests_finish(tiny_model, stub_tokenizer):
    engine = make_engine(tiny_model, stub_tokenizer)
    total = engine.allocator.get_total_blocks()

    engine.generate(["first prompt", "second prompt", "third"], GREEDY)
    # One more step lets the scheduler free the last finished sequence.
    engine.step()

    assert engine.allocator.get_num_free_blocks() == total
    assert engine.allocator.get_num_allocated_blocks() == 0


def test_requests_can_join_while_others_are_running(tiny_model, stub_tokenizer):
    """A late arrival must start generating without waiting for the first to end."""
    engine = make_engine(tiny_model, stub_tokenizer)

    first = engine.submit("early request", SamplingParams(max_new_tokens=12, temperature=0.0))
    for _ in range(3):
        engine.step()
    assert len(engine.runtimes[first].generated) >= 1

    late = engine.submit("late request", SamplingParams(max_new_tokens=4, temperature=0.0))
    engine.step()  # admission + prefill for the late arrival
    engine.step()

    # The late request produced tokens while the first was still unfinished.
    assert len(engine.runtimes[late].generated) >= 1
    assert not engine.runtimes[first].done

    engine.run_until_idle()
    assert engine.runtimes[first].done
    assert engine.runtimes[late].done


def test_sequence_survives_preemption_and_still_completes(tiny_model, stub_tokenizer):
    """
    Under memory pressure the scheduler evicts a sequence, frees its blocks and
    recomputes it later. The final output must still be complete and the pool
    must not leak.
    """
    # A pool too small to hold all three sequences to full length forces eviction.
    engine = make_engine(
        tiny_model, stub_tokenizer, block_size=4, num_blocks=12, max_num_seqs=3
    )
    total = engine.allocator.get_total_blocks()
    params = SamplingParams(max_new_tokens=10, temperature=0.0, top_p=1.0)

    results = engine.generate(
        ["prompt number one", "prompt number two", "prompt number three"], params
    )
    engine.step()

    for result in results:
        assert result["generated_tokens"] == 10, result
    assert engine.total_preemptions > 0, "expected memory pressure to force a preemption"
    assert engine.allocator.get_num_free_blocks() == total, "blocks leaked after preemption"


def test_preempted_output_matches_an_unpressured_run(tiny_model, stub_tokenizer):
    """Recomputation after eviction must not change what the model produces."""
    params = SamplingParams(max_new_tokens=10, temperature=0.0, top_p=1.0)
    prompts = ["prompt number one", "prompt number two", "prompt number three"]

    pressured = make_engine(tiny_model, stub_tokenizer, block_size=4, num_blocks=12, max_num_seqs=3)
    pressured_results = pressured.generate(prompts, params)

    roomy = make_engine(tiny_model, stub_tokenizer, block_size=4, num_blocks=128, max_num_seqs=3)
    roomy_results = roomy.generate(prompts, params)

    assert pressured.total_preemptions > 0
    assert roomy.total_preemptions == 0
    for a, b in zip(pressured_results, roomy_results):
        assert a["token_ids"] == b["token_ids"]


def test_oversized_request_is_rejected_at_submit(tiny_model, stub_tokenizer):
    engine = make_engine(tiny_model, stub_tokenizer, max_model_len=16)
    with pytest.raises(ValueError, match="no room"):
        engine.submit(list(range(20)), GREEDY)


def test_max_new_tokens_is_clamped_to_the_context_window(tiny_model, stub_tokenizer):
    engine = make_engine(tiny_model, stub_tokenizer, max_model_len=20)
    seq_id = engine.submit([1, 2, 3, 4, 5], SamplingParams(max_new_tokens=1000, temperature=0.0))

    assert engine.runtimes[seq_id].params.max_new_tokens == 15


def test_empty_prompt_is_rejected(tiny_model, stub_tokenizer):
    engine = make_engine(tiny_model, stub_tokenizer)
    with pytest.raises(ValueError, match="empty prompt"):
        engine.submit([], GREEDY)


def test_abort_stops_a_sequence_and_frees_its_memory(tiny_model, stub_tokenizer):
    engine = make_engine(tiny_model, stub_tokenizer)
    total = engine.allocator.get_total_blocks()

    seq_id = engine.submit("abort me", SamplingParams(max_new_tokens=50, temperature=0.0))
    engine.step()
    engine.step()
    assert engine.allocator.get_num_allocated_blocks() > 0

    assert engine.abort(seq_id) is True
    engine.step()

    assert engine.runtimes[seq_id].done
    assert engine.allocator.get_num_free_blocks() == total


def test_eos_stops_generation_early(tiny_model, stub_tokenizer):
    """A sequence that emits EOS must stop with finish_reason 'stop'."""
    engine = make_engine(tiny_model, stub_tokenizer)
    seq_id = engine.submit("stop soon", SamplingParams(max_new_tokens=6, temperature=0.0))

    # Drive one step so the sequence exists, then inject EOS through the normal path.
    engine.step()
    runtime = engine.runtimes[seq_id]
    engine._accept_token(runtime, stub_tokenizer.eos_token_id, source="target")

    assert runtime.done
    assert runtime.finish_reason == "stop"
    assert engine.result_for(seq_id)["finish_reason"] == "stop"


def test_seeded_sampling_is_reproducible(tiny_model, stub_tokenizer):
    params = SamplingParams(max_new_tokens=8, temperature=0.9, top_p=0.95, seed=4242)

    first = make_engine(tiny_model, stub_tokenizer).generate("sample me", params)[0]
    second = make_engine(tiny_model, stub_tokenizer).generate("sample me", params)[0]

    assert first["token_ids"] == second["token_ids"]


def test_block_occupancy_maps_blocks_to_their_owners(tiny_model, stub_tokenizer):
    engine = make_engine(tiny_model, stub_tokenizer)
    engine.submit("owner one", SamplingParams(max_new_tokens=6, temperature=0.0))
    engine.submit("owner two", SamplingParams(max_new_tokens=6, temperature=0.0))
    engine.step()
    engine.step()

    occupancy = engine.block_occupancy()

    assert len(occupancy) == engine.allocator.get_total_blocks()
    owners = {owner for owner in occupancy if owner}
    assert owners == {1, 2}
    assert sum(1 for owner in occupancy if owner) == engine.allocator.get_num_allocated_blocks()


def test_telemetry_reports_pool_geometry_and_counters(tiny_model, stub_tokenizer):
    engine = make_engine(tiny_model, stub_tokenizer)
    engine.generate("telemetry check", GREEDY)

    telemetry = engine.telemetry()
    assert telemetry["block_size"] == 4
    assert telemetry["total_blocks"] == 64
    assert telemetry["tokens_generated"] == 8
    assert telemetry["forward_passes"] >= 1
    assert telemetry["steps"] >= 1
    assert telemetry["free_blocks"] + telemetry["allocated_blocks"] == telemetry["total_blocks"]


def test_describe_exposes_the_engine_configuration(tiny_model, stub_tokenizer):
    engine = make_engine(tiny_model, stub_tokenizer)
    described = engine.describe()

    assert described["engine"] == "C++ native (memforge_c)"
    assert described["kv_pool"]["block_size"] == 4
    assert described["kv_pool"]["num_layers"] == tiny_model.config.num_hidden_layers
