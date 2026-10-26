"""
Tests for chunked prefill (Phase C).

The property that matters is that splitting a prompt across steps changes
*scheduling only*: the tokens produced must be identical to prefilling it in one
pass, at every budget. A chunk boundary landing mid-prompt is exactly the sort of
thing that produces fluent, plausible, wrong output if the position bookkeeping
drifts, so the comparisons here are on exact token ids.
"""

from __future__ import annotations

import pytest

from memforge_ml.runtime.continuous_engine import ContinuousBatchingEngine
from memforge_ml.runtime.sampling import SamplingParams

BLOCK = 4


def make_engine(tiny_model, stub_tokenizer, budget, *, sharing=False,
                batched=True, num_blocks=512, max_num_seqs=4):
    return ContinuousBatchingEngine(
        tiny_model,
        stub_tokenizer,
        block_size=BLOCK,
        num_blocks=num_blocks,
        max_num_seqs=max_num_seqs,
        max_model_len=1024,
        enable_prefix_sharing=sharing,
        batched_decode=batched,
        max_prefill_tokens_per_step=budget,
    )


def greedy(tokens=6):
    return SamplingParams(max_new_tokens=tokens, temperature=0.0, stop_on_eos=False)


LONG_PROMPT = "chunked prefill correctness check with a deliberately long prompt " * 2


# ------------------------------------------------------------------ identity

@pytest.mark.parametrize("budget", [64, 32, 16, 8, 4, 1])
def test_chunking_does_not_change_the_output(tiny_model, stub_tokenizer, budget):
    """
    A chunk boundary anywhere in the prompt must be invisible in the output.

    Budget 1 is the extreme case -- one prompt token per step -- and is the most
    likely to expose an off-by-one in the chunk bounds or the position offsets.
    """
    baseline = make_engine(tiny_model, stub_tokenizer, 0).generate(
        [LONG_PROMPT], greedy()
    )[0]["token_ids"]
    chunked = make_engine(tiny_model, stub_tokenizer, budget).generate(
        [LONG_PROMPT], greedy()
    )[0]["token_ids"]
    assert chunked == baseline, f"budget {budget} changed the output"


def test_chunking_composes_with_prefix_sharing(tiny_model, stub_tokenizer):
    """
    Both features advance `num_cached_tokens`, so they share the same cursor.

    If either forgot that the other had moved it, the second prefill chunk would
    start in the wrong place.
    """
    prompts = [LONG_PROMPT + f" tail {i}" for i in range(3)]
    baseline = [
        r["token_ids"]
        for r in make_engine(tiny_model, stub_tokenizer, 0, sharing=True).generate(
            prompts, greedy()
        )
    ]
    chunked = [
        r["token_ids"]
        for r in make_engine(tiny_model, stub_tokenizer, 16, sharing=True).generate(
            prompts, greedy()
        )
    ]
    assert chunked == baseline


def test_chunking_composes_with_sequential_decode(tiny_model, stub_tokenizer):
    baseline = make_engine(tiny_model, stub_tokenizer, 0, batched=False).generate(
        [LONG_PROMPT], greedy()
    )[0]["token_ids"]
    chunked = make_engine(tiny_model, stub_tokenizer, 16, batched=False).generate(
        [LONG_PROMPT], greedy()
    )[0]["token_ids"]
    assert chunked == baseline


def test_short_prompt_under_budget_is_unchunked(tiny_model, stub_tokenizer):
    """A prompt that fits the budget must still take exactly one chunk."""
    engine = make_engine(tiny_model, stub_tokenizer, 256)
    engine.generate(["short"], greedy(3))
    assert engine.telemetry()["prefill_chunks"] == 1


# ------------------------------------------------------------- chunk counting

def test_chunk_count_matches_the_budget(tiny_model, stub_tokenizer, stub_tokenizer_len=None):
    """
    The number of chunks should be ceil(prompt / budget), which is the evidence
    that the budget is actually being applied rather than silently ignored.
    """
    prompt_tokens = len(stub_tokenizer(LONG_PROMPT)["input_ids"])
    for budget in (16, 32):
        engine = make_engine(tiny_model, stub_tokenizer, budget)
        engine.generate([LONG_PROMPT], greedy(2))
        expected = -(-prompt_tokens // budget)  # ceil
        chunks = engine.telemetry()["prefill_chunks"]
        assert chunks == expected, (
            f"budget {budget}: {chunks} chunks for {prompt_tokens} tokens, "
            f"expected {expected}"
        )


def test_telemetry_reports_the_configuration(tiny_model, stub_tokenizer):
    off = make_engine(tiny_model, stub_tokenizer, 0)
    on = make_engine(tiny_model, stub_tokenizer, 128)
    assert off.telemetry()["chunked_prefill_enabled"] is False
    assert on.telemetry()["chunked_prefill_enabled"] is True
    assert on.telemetry()["max_prefill_tokens_per_step"] == 128
    assert on.describe()["chunked_prefill"] is True


# ----------------------------------------------------------- decode fairness

def test_a_long_prompt_does_not_starve_an_in_flight_decode(tiny_model, stub_tokenizer):
    """
    The point of Phase C, measured in steps rather than seconds.

    With chunking on, the in-flight request must emit a token on essentially
    every step while a long prompt is being processed behind it. Counted as
    steps-per-token so the assertion does not depend on machine speed.
    """
    WANT = 12
    engine = make_engine(tiny_model, stub_tokenizer, 8, num_blocks=2048)
    victim = engine.submit("short victim prompt", greedy(WANT))

    # Counted from the very first step, including the warm-up ones: the victim
    # emits a token during its own prefill step too, and not counting those
    # would make the totals disagree with max_new_tokens.
    total_steps = 0
    victim_tokens = 0
    intruder_sent = False
    while engine.has_work() and total_steps < 4000:
        report = engine.step()
        total_steps += 1
        if any(e.seq_id == victim for e in report.events):
            victim_tokens += 1
        # Drop a long prompt on the victim once it is decoding steadily.
        if not intruder_sent and victim_tokens >= 3:
            engine.submit(LONG_PROMPT * 3, greedy(2))
            intruder_sent = True
        if victim_tokens >= WANT:
            break

    assert intruder_sent, "the long prompt was never submitted"
    assert victim_tokens == WANT, f"victim only produced {victim_tokens} tokens"
    # The victim should get a token on essentially every step it is running;
    # a couple of extra steps cover admission and the final chunk boundary.
    assert total_steps <= victim_tokens + 6, (
        f"victim needed {total_steps} steps for {victim_tokens} tokens, so the "
        "long prompt was starving it"
    )


def test_unchunked_prefill_blocks_a_step(tiny_model, stub_tokenizer):
    """
    The behaviour chunking exists to fix, asserted so the comparison is real.

    With no budget, the long prompt is prefilled in one step, and the engine
    needs measurably fewer total steps because it does all that work at once.
    """
    chunked = make_engine(tiny_model, stub_tokenizer, 8, num_blocks=2048)
    chunked.generate([LONG_PROMPT * 2], greedy(4))

    unchunked = make_engine(tiny_model, stub_tokenizer, 0, num_blocks=2048)
    unchunked.generate([LONG_PROMPT * 2], greedy(4))

    assert chunked.telemetry()["prefill_chunks"] > unchunked.telemetry()["prefill_chunks"]
    assert chunked.total_steps > unchunked.total_steps, (
        "chunking should spread the same work over more steps"
    )


# ------------------------------------------------------- scheduler level

def test_scheduler_reports_chunk_bounds():
    """The C++ contract the Python prefill depends on."""
    import memforge_c

    allocator = memforge_c.BlockAllocator(256, 16)
    scheduler = memforge_c.Scheduler(allocator, 4, None, 32)
    assert scheduler.chunked_prefill_enabled() is True
    assert scheduler.get_max_prefill_tokens_per_step() == 32

    scheduler.add_sequence(1, list(range(100)), 4)
    seen = []
    for _ in range(6):
        result = scheduler.step()
        for info in result.running_sequences:
            seen.append((info.num_cached_tokens, info.prefill_chunk_end, info.is_prefill))

    # Chunks must tile the prompt without gaps or overlaps.
    prefills = [(s, e) for s, e, is_pre in seen if is_pre]
    assert prefills[0][0] == 0
    for (_, prev_end), (start, _) in zip(prefills, prefills[1:]):
        assert start == prev_end, f"chunk gap or overlap at {prev_end} -> {start}"
    assert max(e for _, e in prefills) == 100
    # Every chunk respects the budget.
    assert all(e - s <= 32 for s, e in prefills)


def test_scheduler_unlimited_budget_is_one_chunk():
    import memforge_c

    allocator = memforge_c.BlockAllocator(256, 16)
    scheduler = memforge_c.Scheduler(allocator, 4, None, 0)
    assert scheduler.chunked_prefill_enabled() is False

    scheduler.add_sequence(1, list(range(100)), 4)
    result = scheduler.step()
    info = result.running_sequences[0]
    assert info.is_prefill is True
    assert info.num_cached_tokens == 0
    assert info.prefill_chunk_end == 100, "unlimited budget should prefill everything"


def test_budget_is_shared_across_concurrent_prefills():
    """
    One budget for the step, not one per sequence.

    If each sequence got the full budget, N long prompts arriving together would
    multiply the stall by N -- the exact problem chunking is meant to bound.
    """
    import memforge_c

    allocator = memforge_c.BlockAllocator(512, 16)
    scheduler = memforge_c.Scheduler(allocator, 4, None, 48)
    for seq_id in (1, 2, 3):
        scheduler.add_sequence(seq_id, list(range(80)), 4)

    result = scheduler.step()
    chunked = [
        info.prefill_chunk_end - info.num_cached_tokens
        for info in result.running_sequences
        if info.is_prefill
    ]
    assert sum(chunked) <= 48, f"step prefilled {sum(chunked)} tokens on a 48 budget"


def test_zero_length_chunk_is_reported_when_the_budget_runs_out():
    """
    A sequence that gets nothing this step must say so rather than be dropped,
    so the Python layer can skip it and resume next step.
    """
    import memforge_c

    allocator = memforge_c.BlockAllocator(512, 16)
    scheduler = memforge_c.Scheduler(allocator, 4, None, 16)
    for seq_id in (1, 2):
        scheduler.add_sequence(seq_id, list(range(64)), 4)

    result = scheduler.step()
    prefills = [i for i in result.running_sequences if i.is_prefill]
    assert len(prefills) == 2, "both sequences should still be reported as running"
    chunks = sorted(i.prefill_chunk_end - i.num_cached_tokens for i in prefills)
    assert chunks == [0, 16], f"expected one full chunk and one empty, got {chunks}"
