"""
Phase D: speculation and continuous batching composing in one engine.

The phase is "done when speculation and batching compose", so the tests that
matter here are about composition, not about the verification maths (which
`test_batched_speculative.py` covers in isolation):

* **Losslessness at the engine level.** Greedy output with speculation on must
  be *identical* to greedy output with it off, for a batch of concurrently
  decoding sequences. Rejection sampling makes this an equality, not an
  approximation, so `==` on token ids is the right assertion -- and it is a
  strong one, because every one of the moving parts (K+1 block reservation,
  batched verification, partial acceptance, KV write-back, draft rollback) is
  in the path and any of them being wrong changes the tokens.
* **One target pass per round, not one per sequence.** This is the actual
  content of the phase. If verification were per-sequence, speculation would
  cost more target passes than plain batched decoding, which is the reason the
  two could not be used together before.
"""

import pytest

from memforge_ml.runtime.continuous_engine import HAS_CPP_ENGINE, ContinuousBatchingEngine
from memforge_ml.runtime.sampling import SamplingParams

pytestmark = pytest.mark.skipif(
    not HAS_CPP_ENGINE, reason="native memforge_c extension is not built"
)

GREEDY = SamplingParams(max_new_tokens=12, temperature=0.0, top_p=1.0)
PROMPTS = ["alpha prompt", "a much longer beta prompt goes here", "gamma", "delta two"]


def make_engine(model, tokenizer, **kwargs):
    options = dict(block_size=4, num_blocks=256, max_num_seqs=4, max_model_len=256)
    options.update(kwargs)
    return ContinuousBatchingEngine(model, tokenizer, **options)


# ------------------------------------------------------------ losslessness

def test_speculative_batch_output_is_identical_to_plain_batch(
    tiny_model, tiny_draft_model, stub_tokenizer
):
    """
    The headline property: speculation changes the schedule, never the tokens.

    Four sequences decode concurrently, so every round is a genuine batched
    verification with ragged contexts and independent accept/reject decisions.
    """
    plain = make_engine(tiny_model, stub_tokenizer)
    spec = make_engine(
        tiny_model, stub_tokenizer, draft_model=tiny_draft_model, lookahead_k=4
    )

    plain_results = plain.generate(PROMPTS, GREEDY)
    spec_results = spec.generate(PROMPTS, GREEDY)

    assert spec.speculation_enabled
    assert not plain.speculation_enabled

    for prompt, a, b in zip(PROMPTS, plain_results, spec_results):
        assert a["token_ids"] == b["token_ids"], f"speculation changed {prompt!r}"
        assert a["text"] == b["text"]
        assert a["finish_reason"] == b["finish_reason"]


@pytest.mark.parametrize("lookahead", [1, 2, 3, 5, 8])
def test_output_is_independent_of_lookahead(
    tiny_model, tiny_draft_model, stub_tokenizer, lookahead
):
    """
    K is a throughput knob and must not be an output knob.

    Sweeping it also sweeps how often a round ends in a rejection versus a
    bonus token, and crosses the 4-token block boundary at several offsets --
    the arithmetic most likely to be off by one.
    """
    reference = make_engine(tiny_model, stub_tokenizer).generate(PROMPTS, GREEDY)
    spec = make_engine(
        tiny_model, stub_tokenizer,
        draft_model=tiny_draft_model, lookahead_k=lookahead,
    )
    results = spec.generate(PROMPTS, GREEDY)

    for a, b in zip(reference, results):
        assert a["token_ids"] == b["token_ids"], f"K={lookahead} changed the output"


def test_single_sequence_speculation_is_also_lossless(
    tiny_model, tiny_draft_model, stub_tokenizer
):
    """A batch of one still goes through the batched verification path."""
    plain = make_engine(tiny_model, stub_tokenizer).generate(["solo prompt"], GREEDY)
    spec = make_engine(
        tiny_model, stub_tokenizer, draft_model=tiny_draft_model, lookahead_k=3
    ).generate(["solo prompt"], GREEDY)
    assert plain[0]["token_ids"] == spec[0]["token_ids"]


def test_speculation_survives_staggered_arrivals(
    tiny_model, tiny_draft_model, stub_tokenizer
):
    """
    Requests joining mid-flight is the whole point of continuous batching, and
    a late arrival prefills while others are mid-candidate-run. The newcomer
    must still get exactly the tokens it would have got alone.
    """
    engine = make_engine(
        tiny_model, stub_tokenizer, draft_model=tiny_draft_model, lookahead_k=3
    )
    first = engine.submit("early prompt", GREEDY)
    for _ in range(5):
        engine.step()
    late = engine.submit("late arrival prompt", GREEDY)
    engine.run_until_idle()

    alone = make_engine(tiny_model, stub_tokenizer).generate(
        ["early prompt", "late arrival prompt"], GREEDY
    )
    assert engine.result_for(first)["token_ids"] == alone[0]["token_ids"]
    assert engine.result_for(late)["token_ids"] == alone[1]["token_ids"]


# ----------------------------------------------- full acceptance (self-draft)
#
# A draft that *is* the target agrees with it at every greedy step, so every
# candidate is accepted and each round confirms the full K+1. That is the only
# way to reach the multi-token paths on this machine: two independently
# initialised random models agree about once in `vocab_size` tokens, so with
# them every round confirms exactly one token and a cursor that advanced by 1
# instead of by the confirmed count would look perfectly correct.


def self_draft_engine(model, tokenizer, k, **kwargs):
    return make_engine(model, tokenizer, draft_model=model, lookahead_k=k, **kwargs)


def test_self_draft_accepts_every_candidate(tiny_model, stub_tokenizer):
    """Greedy self-speculation accepts all K, so a round confirms K+1."""
    engine = self_draft_engine(tiny_model, stub_tokenizer, 3)
    engine.generate(["alpha prompt"], GREEDY)
    telemetry = engine.telemetry()

    assert telemetry["draft_acceptance_rate"] == 1.0
    assert telemetry["draft_tokens_accepted"] == telemetry["draft_tokens_proposed"]
    # K+1 tokens per round, so far fewer rounds than tokens.
    assert engine.total_tokens_generated >= 3 * telemetry["speculative_rounds"]


@pytest.mark.parametrize("lookahead", [1, 2, 3, 4, 7])
def test_self_draft_output_is_still_exactly_greedy(
    tiny_model, stub_tokenizer, lookahead
):
    """
    With full acceptance, K+1 tokens per round come from one target pass and
    every one of them is written back in a single scatter. The output must
    still be the plain greedy continuation, token for token.

    This is the test that pins the multi-token write-back: advancing the cache
    cursor by anything other than the confirmed count, or scattering fewer K/V
    rows than were confirmed, corrupts the very next round's context and
    changes these tokens.
    """
    reference = make_engine(tiny_model, stub_tokenizer).generate(PROMPTS, GREEDY)
    results = self_draft_engine(tiny_model, stub_tokenizer, lookahead).generate(
        PROMPTS, GREEDY
    )
    for a, b in zip(reference, results):
        assert a["token_ids"] == b["token_ids"], f"K={lookahead} changed the output"


def test_self_draft_confirms_many_tokens_per_round(tiny_model, stub_tokenizer):
    """
    Guards the premise of the tests above: if acceptance ever stopped being
    total, they would silently go back to confirming one token per round and
    stop covering the multi-token paths.
    """
    k = 4
    engine = self_draft_engine(tiny_model, stub_tokenizer, k)
    engine.generate(["alpha prompt", "beta prompt here"], GREEDY)
    rounds = engine.telemetry()["speculative_rounds"]

    assert rounds > 0
    confirmed_per_round = engine.total_tokens_generated / rounds
    assert confirmed_per_round > 1.5, (
        f"only {confirmed_per_round:.2f} tokens per round -- the multi-token "
        f"acceptance path is not being exercised"
    )


def test_self_draft_handles_a_limit_that_is_not_a_multiple_of_the_run(
    tiny_model, stub_tokenizer
):
    """
    `max_new_tokens` that does not divide K+1 forces a sequence to finish
    part-way through an accepted run. The run must be truncated at the limit,
    not rounded up to the round boundary.
    """
    for limit in (1, 2, 5, 7, 9):
        params = SamplingParams(
            max_new_tokens=limit, temperature=0.0, top_p=1.0
        )
        reference = make_engine(tiny_model, stub_tokenizer).generate(PROMPTS, params)
        results = self_draft_engine(tiny_model, stub_tokenizer, 4).generate(
            PROMPTS, params
        )
        for a, b in zip(reference, results):
            assert len(b["token_ids"]) == limit, f"limit {limit} overrun"
            assert a["token_ids"] == b["token_ids"], f"limit {limit} changed output"


def test_self_draft_respects_the_context_limit(tiny_model, stub_tokenizer):
    """
    A sequence approaching `max_model_len` must stop speculating rather than
    reserve candidate slots past the end of its block table.
    """
    engine = self_draft_engine(
        tiny_model, stub_tokenizer, 6, max_model_len=40, num_blocks=256
    )
    params = SamplingParams(max_new_tokens=24, temperature=0.0, top_p=1.0)
    results = engine.generate(["alpha prompt for the context limit"], params)

    reference = make_engine(
        tiny_model, stub_tokenizer, max_model_len=40
    ).generate(["alpha prompt for the context limit"], params)
    assert results[0]["token_ids"] == reference[0]["token_ids"]


def test_self_draft_reclaims_every_block(tiny_model, stub_tokenizer):
    """Full acceptance writes the whole reservation; it must all come back."""
    engine = self_draft_engine(tiny_model, stub_tokenizer, 4)
    free_before = engine.allocator.get_num_free_blocks()
    engine.generate(PROMPTS, GREEDY)
    assert engine.allocator.get_num_free_blocks() == free_before


# --------------------------------------------------------- composition cost

def test_one_target_pass_per_round_not_one_per_sequence(
    tiny_model, tiny_draft_model, stub_tokenizer
):
    """
    The phase's actual claim. Verification is batched, so a round costs one
    target pass regardless of how many sequences are in it.

    Counted directly: `speculative_sequence_rounds` is how many
    (sequence, round) pairs were verified and `speculative_rounds` is how many
    target passes that took. A per-sequence verifier would make those two equal;
    batched verification makes the first strictly larger, and the ratio is the
    number of target passes saved per round.
    """
    engine = make_engine(
        tiny_model, stub_tokenizer, draft_model=tiny_draft_model, lookahead_k=4
    )
    engine.generate(PROMPTS, GREEDY)

    telemetry = engine.telemetry()
    rounds = telemetry["speculative_rounds"]
    assert rounds > 0, "no speculative round ran"

    # More sequences were verified than target passes were spent on them.
    assert engine.total_speculative_sequence_rounds > rounds
    assert telemetry["speculative_sequences_per_round"] > 1.0

    # And the saving is real: a per-sequence verifier would have cost one pass
    # per (sequence, round) pair.
    naive_passes = engine.total_speculative_sequence_rounds
    assert rounds < naive_passes


def test_speculation_confirms_more_than_one_token_per_round(
    tiny_model, tiny_draft_model, stub_tokenizer
):
    """
    A round that only ever confirmed one token would be plain decoding with the
    draft's cost added. With a bonus token always available, tokens generated
    must exceed the number of rounds.
    """
    engine = make_engine(
        tiny_model, stub_tokenizer, draft_model=tiny_draft_model, lookahead_k=4
    )
    engine.generate(PROMPTS, GREEDY)
    telemetry = engine.telemetry()

    assert telemetry["speculative_rounds"] > 0
    assert engine.total_tokens_generated > telemetry["speculative_rounds"]


def test_acceptance_telemetry_is_reported(tiny_model, tiny_draft_model, stub_tokenizer):
    engine = make_engine(
        tiny_model, stub_tokenizer, draft_model=tiny_draft_model, lookahead_k=3
    )
    engine.generate(PROMPTS, GREEDY)
    telemetry = engine.telemetry()

    assert telemetry["speculation_enabled"] is True
    assert telemetry["lookahead_k"] == 3
    proposed = telemetry["draft_tokens_proposed"]
    accepted = telemetry["draft_tokens_accepted"]
    assert proposed > 0
    assert 0 <= accepted <= proposed
    assert telemetry["draft_acceptance_rate"] == pytest.approx(accepted / proposed)


def test_telemetry_reports_speculation_off_by_default(tiny_model, stub_tokenizer):
    telemetry = make_engine(tiny_model, stub_tokenizer).telemetry()
    assert telemetry["speculation_enabled"] is False
    assert telemetry["lookahead_k"] == 0
    assert telemetry["draft_acceptance_rate"] == 0.0


def test_eos_part_way_through_an_accepted_run(tiny_model, stub_tokenizer):
    """
    A sequence can stop on EOS at any position inside an accepted run, and the
    run must be cut there exactly.

    This is the one case the headroom check cannot pre-empt: it knows how many
    tokens a sequence is allowed, but not which token the model will produce.
    EOS is forced here by declaring the model's own third greedy token to be
    EOS, because a randomly initialised model will not emit the real one.

    Without the cut, the tokens after EOS would still be emitted -- visible to a
    streaming client as text past the end of the response, while
    `generated_tokens` and the stop reason looked correct.
    """
    k = 4
    prompt = "alpha prompt"
    # Prefill emits the first token, so a sequence only has headroom to
    # speculate while `1 + (K+1) <= max_new_tokens`. At `max_new_tokens = K+1`
    # it never speculates at all and this test silently measures the plain
    # decode path -- which is exactly what it did before the bound below was
    # added. Hence a generous limit and an explicit round-count assertion.
    params = SamplingParams(max_new_tokens=12, temperature=0.0, top_p=1.0)

    # What the model does with no stopping at all.
    baseline = make_engine(tiny_model, stub_tokenizer).generate([prompt], params)
    tokens = baseline[0]["token_ids"]

    # The first speculative round covers output indices 1..K+1 (index 0 came
    # from prefill), so index 3 is strictly inside it with tokens still to come.
    cut = 3
    eos_token = tokens[cut]
    assert eos_token not in tokens[:cut], "pick an EOS that first occurs mid-run"

    engine = self_draft_engine(tiny_model, stub_tokenizer, k)
    engine.eos_token_ids = {eos_token}
    seq_id = engine.submit(prompt, params)

    # Bounded on purpose. If the run is not cut at EOS, the sequence keeps
    # being fed tokens, is never retired, and the engine spins to `max_steps`;
    # unbounded, that reads in CI as a hung job rather than a failure. One
    # prefill plus one round is enough, so a handful of steps is generous.
    reports = engine.run_until_idle(max_steps=8)
    assert len(reports) < 8, "sequence was not retired after stopping on EOS"

    # Non-vacuity: a speculative round must actually have run, or the EOS was
    # cut out of an ordinary decode step and this test proves nothing.
    assert engine.telemetry()["speculative_rounds"] > 0, "no speculative round ran"

    result = engine.result_for(seq_id)

    # Stopped at EOS inclusive, and nothing from the rest of the run leaked out.
    assert result["token_ids"] == tokens[: cut + 1]
    assert result["finish_reason"] == "stop"

    # The event stream is what a streaming client sees, and it must agree with
    # the final result rather than carrying the discarded tail.
    emitted = [ev.token_id for ev in engine.drain_events(seq_id)]
    assert emitted == tokens[: cut + 1]


def test_headroom_check_avoids_drafting_that_cannot_be_used(
    tiny_model, stub_tokenizer
):
    """
    A sequence with fewer than K+1 tokens left must not run the draft at all:
    every candidate it produced would be discarded.

    Asserted as a round count. With `max_new_tokens = K + 3` there is room for
    exactly one full run, after which the remainder decodes plainly -- so one
    speculative round. Dropping the check yields two, the second of which
    drafts K tokens to use at most 2.
    """
    k = 4
    params = SamplingParams(max_new_tokens=k + 3, temperature=0.0, top_p=1.0)
    engine = self_draft_engine(tiny_model, stub_tokenizer, k)
    engine.generate(["alpha prompt"], params)

    assert engine.telemetry()["speculative_rounds"] == 1
    assert engine.total_tokens_generated == k + 3


def test_speculative_generation_terminates_in_bounded_steps(
    tiny_model, stub_tokenizer
):
    """
    A speculative run must finish in a predictable number of steps.

    Each round confirms at most K+1 tokens, so `max_new_tokens / (K+1)` rounds
    plus a prefill step per sequence is the floor; the budget here is several
    times that. The test exists because the failure it guards against is
    unbounded rather than wrong: when a finished sequence keeps being fed
    tokens it is never retired, `has_work()` stays true, and the engine spins
    until `max_steps`. Asserting on output alone cannot see that -- the tokens
    come out right, just hours later.
    """
    k = 4
    tokens_wanted = 12
    params = SamplingParams(
        max_new_tokens=tokens_wanted, temperature=0.0, top_p=1.0
    )
    engine = self_draft_engine(tiny_model, stub_tokenizer, k)
    seq_ids = [engine.submit(prompt, params) for prompt in PROMPTS]

    budget = 40
    reports = engine.run_until_idle(max_steps=budget)

    assert len(reports) < budget, (
        f"engine still had work after {budget} steps; sequences are not "
        f"being retired"
    )
    assert not engine.has_work()
    for seq_id in seq_ids:
        result = engine.result_for(seq_id)
        assert result["finish_reason"] is not None
        assert len(result["token_ids"]) == tokens_wanted


# ------------------------------------------------------------- degradation

def test_draft_model_without_lookahead_stays_disabled(
    tiny_model, tiny_draft_model, stub_tokenizer
):
    """Speculation with K=0 is plain decoding, so it must not engage."""
    engine = make_engine(
        tiny_model, stub_tokenizer, draft_model=tiny_draft_model, lookahead_k=0
    )
    assert not engine.speculation_enabled
    engine.generate(["hello"], GREEDY)
    assert engine.telemetry()["speculative_rounds"] == 0


def test_lookahead_without_draft_model_stays_disabled(tiny_model, stub_tokenizer):
    engine = make_engine(tiny_model, stub_tokenizer, lookahead_k=4)
    assert not engine.speculation_enabled


def test_sequence_near_its_token_limit_falls_back_cleanly(
    tiny_model, tiny_draft_model, stub_tokenizer
):
    """
    A sequence with fewer than K+1 tokens left cannot take a full candidate
    run, so it decodes normally *in the same step* as its neighbours speculate.
    Its output must be unaffected, which is what makes the fallback safe rather
    than merely non-crashing.
    """
    tight = SamplingParams(max_new_tokens=3, temperature=0.0, top_p=1.0)
    plain = make_engine(tiny_model, stub_tokenizer).generate(PROMPTS, tight)
    spec = make_engine(
        tiny_model, stub_tokenizer, draft_model=tiny_draft_model, lookahead_k=6
    )
    results = spec.generate(PROMPTS, tight)

    for a, b in zip(plain, results):
        assert a["token_ids"] == b["token_ids"]
        assert len(b["token_ids"]) == 3


def test_scheduler_reserves_the_candidate_slots(
    tiny_model, tiny_draft_model, stub_tokenizer
):
    """
    The engine must hand K+1 to the C++ scheduler, not K and not 1. Without the
    extra slot, the bonus token has nowhere to be written on the one round in
    which every candidate was accepted -- a rare, load-dependent failure.
    """
    engine = make_engine(
        tiny_model, stub_tokenizer, draft_model=tiny_draft_model, lookahead_k=4
    )
    assert engine.scheduler.get_lookahead_slots() == 5

    plain = make_engine(tiny_model, stub_tokenizer)
    assert plain.scheduler.get_lookahead_slots() == 0


def test_blocks_are_fully_reclaimed_after_speculative_generation(
    tiny_model, tiny_draft_model, stub_tokenizer
):
    """
    Reserved-but-rejected candidate blocks must come back. A leak here would
    only show up as a slow starvation under sustained load.
    """
    engine = make_engine(
        tiny_model, stub_tokenizer, draft_model=tiny_draft_model, lookahead_k=4
    )
    free_before = engine.allocator.get_num_free_blocks()
    engine.generate(PROMPTS, GREEDY)
    assert engine.allocator.get_num_free_blocks() == free_before


# ----------------------------------------------------------- server wiring

def test_server_config_exposes_batched_speculation_off_by_default():
    """
    Off by default is a deliberate choice, not an oversight: the feature is
    lossless but measured slower on this hardware, so defaulting it on would
    make the server worse out of the box.
    """
    from server.config import ServerConfig

    assert ServerConfig().batched_speculation is False


def test_server_config_reads_the_env_var(monkeypatch):
    """
    `ServerConfig` fields are evaluated at class-definition time, so the module
    has to be reimported for a new environment to take effect -- the same
    restart-required behaviour the onboarding doc warns about.
    """
    import importlib

    import server.config

    monkeypatch.setenv("MEMFORGE_BATCHED_SPECULATION", "true")
    monkeypatch.setenv("MEMFORGE_LOOKAHEAD_K", "6")
    reloaded = importlib.reload(server.config)
    try:
        cfg = reloaded.ServerConfig()
        assert cfg.batched_speculation is True
        assert cfg.lookahead_k == 6
    finally:
        # Leave the module as the rest of the suite expects to find it.
        monkeypatch.delenv("MEMFORGE_BATCHED_SPECULATION")
        monkeypatch.delenv("MEMFORGE_LOOKAHEAD_K")
        importlib.reload(server.config)


def test_engine_reaches_the_scheduler_with_the_configured_lookahead(
    tiny_model, tiny_draft_model, stub_tokenizer
):
    """
    The config value has to arrive at the C++ scheduler as K+1, not stop at the
    Python layer. This asserts the whole chain in one step, because each link
    looked right in isolation while the feature did nothing.
    """
    engine = make_engine(
        tiny_model, stub_tokenizer, draft_model=tiny_draft_model, lookahead_k=6
    )
    assert engine.lookahead_k == 6
    assert engine.scheduler.get_lookahead_slots() == 7
    assert engine.telemetry()["lookahead_k"] == 6
