"""
Prefill/decode fusion in the engine (Phase C bullet 2).

The claim is narrow and checkable: when a step carries both a prefill chunk and
decode tokens, running them as one packed pass must produce the same tokens as
running them as two passes. Everything else about the feature -- whether it is
faster, by how much -- is a benchmark's job; these tests are only about it being
*the same answer*.

Why token equality and not logit equality. The varlen kernel accumulates an
online softmax in fp32 over blocks, while the unfused path runs transformers'
own attention over a materialised context. Those sum in different orders, so
bit-identical logits are not available and demanding them would be testing the
wrong thing. Greedy token ids are discrete and are compared exactly; that is
the property a user can observe.

The `bound.used` check inside the engine matters as much as these tests: a layer
that quietly fell back to eager attention would use transformers' causal mask,
which for a packed run lets one sequence attend to another's keys. That is a
cross-request KV leak that reads as fluent text, so the engine raises rather
than falls back, and `test_every_layer_takes_the_varlen_path` pins it.
"""

from __future__ import annotations

import pytest

from memforge_ml.runtime.continuous_engine import HAS_CPP_ENGINE, ContinuousBatchingEngine
from memforge_ml.runtime.sampling import SamplingParams

pytestmark = pytest.mark.skipif(
    not HAS_CPP_ENGINE, reason="native memforge_c extension is not built"
)

GREEDY = SamplingParams(max_new_tokens=10, temperature=0.0, top_p=1.0, stop_on_eos=False)


def make_engine(model, tokenizer, **kwargs):
    options = dict(
        block_size=4, num_blocks=512, max_num_seqs=4, max_model_len=512,
    )
    options.update(kwargs)
    return ContinuousBatchingEngine(model, tokenizer, **options)


def long_prompt(n: int) -> str:
    """A prompt of roughly `n` tokens under the stub tokenizer (1 char = 1 token)."""
    return "x" * n


def run_staggered(engine, warmup_steps: int = 3):
    """
    A short request decoding steadily, then a long prompt landing on it.

    This is the traffic pattern that produces mixed steps: once the long prompt
    is admitted and chunked, each step carries a chunk *and* the first
    sequence's decode token, which is exactly what fusion is for.
    """
    victim = engine.submit("hello there", GREEDY)
    for _ in range(warmup_steps):
        engine.step()
    intruder = engine.submit(long_prompt(60), GREEDY)
    engine.run_until_idle(max_steps=4000)
    return victim, intruder


# ------------------------------------------------------- fusion == no fusion

@pytest.mark.parametrize("budget", [8, 16, 32])
def test_fused_output_matches_unfused(tiny_model, stub_tokenizer, budget):
    """
    The headline property, across prefill budgets. A budget that does not
    divide the prompt leaves a short final chunk, which is where chunk-boundary
    arithmetic goes wrong.
    """
    plain = make_engine(
        tiny_model, stub_tokenizer, max_prefill_tokens_per_step=budget
    )
    fused = make_engine(
        tiny_model, stub_tokenizer,
        max_prefill_tokens_per_step=budget, fused_prefill=True,
    )
    assert fused.fused_prefill, "fusion did not install; the test would be vacuous"

    plain_ids = run_staggered(plain)
    fused_ids = run_staggered(fused)

    # Non-vacuity: at least one step must actually have fused, or this is just
    # two runs of the same code path.
    assert fused.telemetry()["fused_steps"] > 0, "no step was fused"

    for which, p, f in zip(("victim", "intruder"), plain_ids, fused_ids):
        want = plain.result_for(p)
        got = fused.result_for(f)
        assert got["token_ids"] == want["token_ids"], (
            f"{which}: fusion changed the output\n"
            f"  unfused {want['token_ids']}\n  fused   {got['token_ids']}"
        )
        assert got["finish_reason"] == want["finish_reason"]


def test_fusion_saves_a_forward_pass_per_mixed_step(tiny_model, stub_tokenizer):
    """
    The point of the feature: a mixed step costs one pass instead of two.

    Asserted as a pass count rather than a timing, because a count is exact and
    a CPU timing on a tiny model is noise. The saving must be at least one pass
    per fused step.
    """
    plain = make_engine(tiny_model, stub_tokenizer, max_prefill_tokens_per_step=16)
    fused = make_engine(
        tiny_model, stub_tokenizer,
        max_prefill_tokens_per_step=16, fused_prefill=True,
    )
    run_staggered(plain)
    run_staggered(fused)

    fused_steps = fused.telemetry()["fused_steps"]
    assert fused_steps > 0
    assert fused.total_forward_passes <= plain.total_forward_passes - fused_steps, (
        f"fused ran {fused.total_forward_passes} passes against "
        f"{plain.total_forward_passes} unfused, over {fused_steps} fused steps"
    )


def test_fusion_is_off_by_default(tiny_model, stub_tokenizer):
    engine = make_engine(tiny_model, stub_tokenizer)
    assert engine.fused_prefill is False
    telemetry = engine.telemetry()
    assert telemetry["fused_prefill_enabled"] is False
    assert telemetry["fused_steps"] == 0


def test_fusion_needs_both_kinds_of_work(tiny_model, stub_tokenizer):
    """
    A step with only prefills, or only decodes, must take the ordinary path.
    Fusing one kind of work would be the same pass with extra bookkeeping.
    """
    engine = make_engine(
        tiny_model, stub_tokenizer,
        max_prefill_tokens_per_step=16, fused_prefill=True,
    )
    # A single request: its prefill step has no decode alongside it, and every
    # later step is decode-only.
    engine.generate(["solo prompt"], GREEDY)
    assert engine.telemetry()["fused_steps"] == 0


def test_fused_tokens_counts_every_packed_token(tiny_model, stub_tokenizer):
    engine = make_engine(
        tiny_model, stub_tokenizer,
        max_prefill_tokens_per_step=16, fused_prefill=True,
    )
    run_staggered(engine)
    telemetry = engine.telemetry()
    # Every fused step packs at least one chunk token plus one decode token.
    assert telemetry["fused_tokens"] >= 2 * telemetry["fused_steps"]


# ------------------------------------------------------------- the guard rail

def test_every_layer_takes_the_varlen_path(tiny_model, stub_tokenizer, monkeypatch):
    """
    A silent fallback in a fused pass is a cross-sequence KV leak, so the engine
    refuses rather than degrading. This forces a fallback and asserts it raises.
    """
    from memforge_ml.kernels import attention_backend

    engine = make_engine(
        tiny_model, stub_tokenizer,
        max_prefill_tokens_per_step=16, fused_prefill=True,
    )

    # Make the varlen binding invisible to the attention function, which is
    # exactly what a layer bypassing the backend would look like.
    monkeypatch.setattr(attention_backend, "_current_varlen", lambda: None)

    with pytest.raises(RuntimeError, match="layers used"):
        run_staggered(engine)


def test_fusion_rejects_the_decode_mirror(tiny_model, stub_tokenizer):
    """
    The mirror never sees a packed write, so combining the two would serve
    stale KV. Refused at construction rather than silently preferring one.
    """
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_engine(
            tiny_model, stub_tokenizer,
            fused_prefill=True, decode_view_cache=True,
        )


# ---------------------------------------------------- interaction with Phase B

def test_fusion_still_shares_prefixes(tiny_model, stub_tokenizer):
    """
    A fused prefill must still register its blocks, or enabling fusion would
    quietly switch prefix sharing off.

    Two things this test has to get right, both of which it got wrong first:

      * **The metric.** `cached_prefix_tokens` counts chunked-prefill
        resumption as well as cache hits -- a 60-token prompt at budget 16
        resumes at 16, 32 and 48, reporting 96 "cached" tokens with nothing
        shared at all. So it is satisfied whether or not registration happens.
        `prefix_cache_hits` is the figure that means sharing.
      * **The overlap.** The twin prompts must be in flight at the same time. A
        finished sequence's blocks return to the allocator and are evicted from
        the registry -- that is exactly the mechanism that prevents stale reuse
        -- so running the first to completion leaves nothing to share.
    """
    long_lived = SamplingParams(
        max_new_tokens=60, temperature=0.0, top_p=1.0, stop_on_eos=False
    )
    shared = long_prompt(60)

    engine = make_engine(
        tiny_model, stub_tokenizer,
        max_prefill_tokens_per_step=16, fused_prefill=True,
    )
    # A long-lived decode keeps every prefill step mixed, which is what forces
    # the fused path.
    engine.submit("aaa", long_lived)
    for _ in range(3):
        engine.step()

    first = engine.submit(shared, long_lived)
    # Let the first copy finish prefilling and start decoding, so its blocks are
    # registered and still referenced.
    for _ in range(40):
        engine.step()
    assert engine.telemetry()["fused_steps"] > 0, "no fused step; test is vacuous"
    hits_before = engine.telemetry()["prefix_cache_hits"]

    engine.submit(shared, long_lived)
    engine.run_until_idle(max_steps=4000)

    gained = engine.telemetry()["prefix_cache_hits"] - hits_before
    assert gained > 0, (
        "the second copy of an identical 60-token prompt scored no prefix-cache "
        "hit, so fused prefill is not registering its blocks"
    )
    assert engine.result_for(first)["token_ids"]

def test_blocks_are_reclaimed_after_fused_steps(tiny_model, stub_tokenizer):
    engine = make_engine(
        tiny_model, stub_tokenizer,
        max_prefill_tokens_per_step=16, fused_prefill=True,
    )
    free_before = engine.allocator.get_num_free_blocks()
    run_staggered(engine)
    assert engine.telemetry()["fused_steps"] > 0
    assert engine.allocator.get_num_free_blocks() == free_before


# ------------------------------------------------- the backend stays scoped

def test_fused_pass_restores_the_models_attention_implementation(
    tiny_model, stub_tokenizer
):
    """
    Rerouting a model's attention must not outlive the fused pass.

    This was a real bug: the engine called `enable_on` once at construction,
    which left `_attn_implementation` pointing at this backend forever. Every
    *ordinary* decode then went through it, found no binding, and fell back to
    the backend's own eager attention instead of the attention the model was
    configured with -- changing numerics for work that had no reason to change,
    and leaking across every engine sharing the model. It surfaced as twelve
    unrelated speculative-decoding tests failing, because the model fixture is
    shared for the whole session.
    """
    from memforge_ml.kernels import attention_backend

    before = tiny_model.config._attn_implementation
    engine = make_engine(
        tiny_model, stub_tokenizer,
        max_prefill_tokens_per_step=16, fused_prefill=True,
    )
    assert tiny_model.config._attn_implementation == before, (
        "constructing the engine rerouted the model's attention"
    )

    run_staggered(engine)
    assert engine.telemetry()["fused_steps"] > 0, "no fused pass ran"
    assert tiny_model.config._attn_implementation == before, (
        "a fused pass left the model rerouted to "
        f"{tiny_model.config._attn_implementation}"
    )
    assert not attention_backend.is_enabled_on(tiny_model)


def test_an_unfused_engine_on_the_same_model_is_unaffected(
    tiny_model, stub_tokenizer
):
    """
    Two engines sharing one model must not influence each other's numerics.
    The fused engine's output is checked against an engine built *after* it,
    so any lingering reroute would show up as a token mismatch.
    """
    fused = make_engine(
        tiny_model, stub_tokenizer,
        max_prefill_tokens_per_step=16, fused_prefill=True,
    )
    run_staggered(fused)
    assert fused.telemetry()["fused_steps"] > 0

    after = make_engine(tiny_model, stub_tokenizer, max_prefill_tokens_per_step=16)
    before = make_engine(tiny_model, stub_tokenizer, max_prefill_tokens_per_step=16)
    a_ids = run_staggered(after)
    b_ids = run_staggered(before)
    for a, b in zip(a_ids, b_ids):
        assert after.result_for(a)["token_ids"] == before.result_for(b)["token_ids"]


# ----------------------------------------------- exact cursor invariants

def fused_step_once(engine, prompt_tokens=60, budget=16, warmup=3):
    """
    Drive the engine to exactly one fused step and stop.

    Returns `(victim_runtime, intruder_runtime, decode_position_before)`, so a
    caller can assert on the cursors the fused step was responsible for moving.
    """
    victim = engine.submit("hello there", GREEDY)
    for _ in range(warmup):
        engine.step()
    intruder = engine.submit(long_prompt(prompt_tokens), GREEDY)

    fused_before = engine.telemetry()["fused_steps"]
    for _ in range(200):
        v_rt = engine.runtimes[victim]
        position_before = v_rt.cache.position if v_rt.cache is not None else None
        engine.step()
        if engine.telemetry()["fused_steps"] > fused_before:
            return engine.runtimes[victim], engine.runtimes[intruder], position_before
    raise AssertionError("no fused step occurred within 200 steps")


def test_fused_step_advances_the_prefill_cursor_to_the_chunk_end(
    tiny_model, stub_tokenizer
):
    """
    After a fused chunk, the prefilling sequence's cursor must sit at the end of
    what was written. Leaving it at the chunk *start* makes the next chunk
    overwrite this one, and leaving it unmoved makes every chunk land on the
    same slots -- both of which only show up later, as wrong KV.
    """
    budget = 16
    engine = make_engine(
        tiny_model, stub_tokenizer,
        max_prefill_tokens_per_step=budget, fused_prefill=True,
    )
    _, intruder, _ = fused_step_once(engine, budget=budget)

    # The first fused chunk starts at 0, so the cursor must be exactly the
    # number of tokens the budget allowed.
    assert intruder.cache is not None
    assert intruder.cache.position == budget, (
        f"prefill cursor is at {intruder.cache.position}, expected {budget}"
    )
    # Every layer must agree; a per-layer cursor that drifted would corrupt one
    # layer's reads only, which is the hardest kind of bug to see.
    assert {layer.position for layer in intruder.cache.layers} == {budget}


def test_fused_step_advances_the_decode_cursor_by_exactly_one(
    tiny_model, stub_tokenizer
):
    """
    A decode token in a fused step writes one slot and must move the cursor one
    place. Not moving it makes the next token overwrite this one and re-read a
    context short by one -- fluent, confident, wrong.
    """
    engine = make_engine(
        tiny_model, stub_tokenizer,
        max_prefill_tokens_per_step=16, fused_prefill=True,
    )
    victim, _, before = fused_step_once(engine)
    assert before is not None
    assert victim.cache is not None
    assert victim.cache.position == before + 1, (
        f"decode cursor went {before} -> {victim.cache.position}, expected "
        f"{before + 1}"
    )
    assert {layer.position for layer in victim.cache.layers} == {before + 1}


# --------------------------------------------- logit-level equivalence

def test_fused_logits_match_the_unfused_logits(tiny_model, stub_tokenizer):
    """
    The sensitive version of the headline test.

    Sampled tokens are a coarse signal -- a wrong RoPE position can shift every
    logit and still leave the argmax alone over a small vocabulary, and that is
    exactly what let a mutation replacing absolute positions with the packing
    index survive. Comparing the logits themselves catches it.

    Both engines are driven to the same step with the same traffic, and the
    fused engine's packed logits are compared row by row against the unfused
    engine's two separate passes.
    """
    import torch

    budget = 16
    captured = {}

    def capture(engine, key):
        original = engine.model.forward

        def wrapper(*args, **kwargs):
            out = original(*args, **kwargs)
            captured.setdefault(key, []).append(out.logits.detach().clone())
            return out

        engine.model.forward = wrapper
        return original

    fused = make_engine(
        tiny_model, stub_tokenizer,
        max_prefill_tokens_per_step=budget, fused_prefill=True,
    )
    plain = make_engine(
        tiny_model, stub_tokenizer, max_prefill_tokens_per_step=budget
    )

    # Run both to the same point, then capture only the step of interest.
    for engine, key in ((fused, "fused"), (plain, "plain")):
        victim = engine.submit("hello there", GREEDY)
        for _ in range(3):
            engine.step()
        engine.submit(long_prompt(60), GREEDY)
        restore = capture(engine, key)
        try:
            engine.step()
        finally:
            engine.model.forward = restore

    assert fused.telemetry()["fused_steps"] == 1, "the captured step was not fused"
    assert len(captured["fused"]) == 1, "a fused step must be one forward pass"
    assert len(captured["plain"]) == 2, "the unfused step should be two passes"

    fused_logits = captured["fused"][0]          # (1, chunk + 1, vocab)
    chunk_logits, decode_logits = captured["plain"]

    chunk_width = chunk_logits.shape[1]
    assert fused_logits.shape[1] == chunk_width + 1, (
        f"packed {fused_logits.shape[1]} rows for a {chunk_width}-token chunk "
        f"plus one decode token"
    )

    # The prefill chunk's rows.
    delta = (fused_logits[0, :chunk_width] - chunk_logits[0]).abs().max()
    assert delta < 2e-4, f"fused chunk logits diverged by {delta:.3e}"
    # The decode token's row, which is where a wrong absolute position shows up:
    # it sits at position ~13 while its packing index is ~16.
    delta = (fused_logits[0, chunk_width] - decode_logits[0, -1]).abs().max()
    assert delta < 2e-4, f"fused decode logits diverged by {delta:.3e}"


def test_a_fused_final_chunk_registers_its_prefix(tiny_model, stub_tokenizer):
    """
    Registration happens only on a prompt's *final* chunk, so it is only the
    fused path's job when that last chunk is itself fused.

    Without a long-lived neighbour the short warm-up request finishes before the
    long prompt's last chunk, that chunk runs through the ordinary
    `_run_prefill`, and deleting the registration from the fused path changes
    nothing. The neighbour here is given enough tokens to still be decoding at
    the final chunk, which is what puts the fused branch on the hook.
    """
    budget = 16
    prompt_tokens = 60
    chunks = -(-prompt_tokens // budget)  # 4
    long_lived = SamplingParams(
        max_new_tokens=90, temperature=0.0, top_p=1.0, stop_on_eos=False
    )
    shared = long_prompt(prompt_tokens)

    engine = make_engine(
        tiny_model, stub_tokenizer,
        max_prefill_tokens_per_step=budget, fused_prefill=True,
    )
    engine.submit("aaa", long_lived)
    for _ in range(3):
        engine.step()

    engine.submit(shared, long_lived)
    for _ in range(40):
        engine.step()

    # Every chunk shared a step with the long-lived decode, so the final one was
    # fused and only the fused path can have registered the prefix.
    assert engine.telemetry()["fused_steps"] >= chunks, (
        f"only {engine.telemetry()['fused_steps']} fused steps for {chunks} "
        f"chunks; the final chunk was not fused and this test is vacuous"
    )
    hits_before = engine.telemetry()["prefix_cache_hits"]

    engine.submit(shared, long_lived)
    engine.run_until_idle(max_steps=4000)

    gained = engine.telemetry()["prefix_cache_hits"] - hits_before
    assert gained > 0, (
        "the repeated prompt scored no prefix-cache hit; the fused final chunk "
        "did not register its blocks"
    )
