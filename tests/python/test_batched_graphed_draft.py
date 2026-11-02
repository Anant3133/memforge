"""
Batched CUDA-graph draft: one replay drafts every running sequence.

Like the per-sequence graph it must change speed only: greedy tokens equal plain
decoding AND equal speculation with the eager draft, and the *proposals* (the
token sources) must equal the eager draft's, since lossless output alone cannot
reveal a broken draft. Sequences have different lengths, so rows sit at
different positions inside one replay.
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

from memforge_ml.runtime.continuous_engine import HAS_CPP_ENGINE, ContinuousBatchingEngine
from memforge_ml.runtime.sampling import SamplingParams

TARGET, DRAFT = "HuggingFaceTB/SmolLM2-360M-Instruct", "HuggingFaceTB/SmolLM2-135M-Instruct"
PROMPTS = ["The capital of France is", "Once upon a time, in a land far away, there lived a",
           "def fibonacci(n):", "Explain why the sky is blue in one sentence, briefly."]
GREEDY = SamplingParams(max_new_tokens=24, temperature=0.0, top_p=1.0, stop_on_eos=False)
pytestmark = [pytestmark, pytest.mark.skipif(not HAS_CPP_ENGINE, reason="native extension not built")]


@pytest.fixture(scope="module")
def models():
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(TARGET, local_files_only=True)
        t = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.float32, local_files_only=True)
        d = AutoModelForCausalLM.from_pretrained(DRAFT, dtype=torch.float32, local_files_only=True)
    except OSError:
        pytest.skip("models not cached")
    return tok, t.cuda().eval(), d.cuda().eval()


def run(models, prompts=PROMPTS, params=GREEDY, slots=4, **kw):
    tok, t, _ = models
    eng = ContinuousBatchingEngine(t, tok, block_size=16, kv_budget_mb=256, max_num_seqs=slots,
                                   max_model_len=512, **kw)
    res = eng.generate(prompts, params)
    return eng, [r["token_ids"] for r in res], [r["token_sources"] for r in res]


def test_batched_graph_matches_plain_and_eager_draft(models):
    d = models[2]
    _, plain, _ = run(models)
    _, eager, eager_src = run(models, draft_model=d, lookahead_k=4)
    eng, got, src = run(models, draft_model=d, lookahead_k=4, draft_cuda_graph="batched")
    assert got == plain == eager, "batched graphed speculation changed the tokens"
    assert src == eager_src, "batched graphed draft proposes different tokens than the eager draft"
    assert any("draft" in str(x) for seq in src for x in seq), "nothing was ever accepted"
    assert len(eng._draft_pool._free) == 4 and eng._draft_pool._graph


@pytest.mark.parametrize("n_prompts,slots", [(1, 4), (2, 4), (3, 8), (7, 4)])
def test_any_batch_size_including_more_requests_than_slots(models, n_prompts, slots):
    d = models[2]
    prompts = (PROMPTS * 2)[:n_prompts]
    _, plain, _ = run(models, prompts, slots=slots)
    eng, got, _ = run(models, prompts, slots=slots, draft_model=d, lookahead_k=3,
                      draft_cuda_graph="batched")
    assert got == plain
    assert len(eng._draft_pool._free) == slots


def test_one_graph_per_round_not_per_sequence(models):
    """The point of the change: replays per round do not grow with the batch."""
    d = models[2]
    eng, _, _ = run(models, draft_model=d, lookahead_k=3, draft_cuda_graph="batched")
    pool = eng._draft_pool
    calls = []
    for b, g in list(pool._graph.items()):
        orig = g.replay
        g.replay = (lambda o=orig: (calls.append(1), o())[1])
    eng.generate(PROMPTS, GREEDY)
    rounds = eng.total_speculative_rounds
    assert calls, "no graph replays happened"
    # (K+1) replays per round for the whole batch (second generate adds rounds)
    per_round = len(calls) / max(1, rounds - rounds // 2)
    assert per_round <= 3 + 1 + 0.5, f"{per_round} replays/round; expected ~K+1=4"


def test_sampling_runs_and_frees_slots(models):
    d = models[2]
    params = SamplingParams(max_new_tokens=16, temperature=0.8, top_p=1.0, stop_on_eos=False)
    eng, got, _ = run(models, params=params, draft_model=d, lookahead_k=3, draft_cuda_graph="batched")
    assert all(len(g) == 16 for g in got) and len(eng._draft_pool._free) == 4


@pytest.mark.parametrize("n_prompts,slots", [(1, 4), (3, 4), (4, 8), (7, 4)])
def test_unrolled_one_replay_per_round_matches_eager(models, n_prompts, slots):
    d = models[2]
    prompts = (PROMPTS * 2)[:n_prompts]
    _, plain, _ = run(models, prompts, slots=slots)
    _, eager, esrc = run(models, prompts, slots=slots, draft_model=d, lookahead_k=4)
    eng, got, src = run(models, prompts, slots=slots, draft_model=d, lookahead_k=4,
                        draft_cuda_graph="batched-unrolled")
    assert got == plain == eager
    assert src == esrc, "unrolled draft proposes different tokens than the eager draft"
    assert eng._draft_pool._ugraph, "the unrolled graph never ran"
    assert len(eng._draft_pool._free) == slots


def test_unrolled_falls_back_to_stepwise_when_sampling(models):
    d = models[2]
    params = SamplingParams(max_new_tokens=12, temperature=0.8, top_p=1.0, stop_on_eos=False)
    eng, got, _ = run(models, params=params, draft_model=d, lookahead_k=3,
                      draft_cuda_graph="batched-unrolled")
    assert all(len(g) == 12 for g in got) and not eng._draft_pool._ugraph
