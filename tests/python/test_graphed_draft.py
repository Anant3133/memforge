"""
CUDA-graph draft: must change the schedule's *speed*, never its tokens.

Greedy output with a graph-replayed draft has to equal plain greedy decoding AND
speculation with the eager draft, token for token, across concurrent sequences
with different lengths (so rollback, catch-up prefill and slot reuse all run).
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

from memforge_ml.runtime.continuous_engine import HAS_CPP_ENGINE, ContinuousBatchingEngine
from memforge_ml.runtime.sampling import SamplingParams

TARGET, DRAFT = "HuggingFaceTB/SmolLM2-360M-Instruct", "HuggingFaceTB/SmolLM2-135M-Instruct"
PROMPTS = ["The capital of France is", "Once upon a time, in a land far away,",
           "def fibonacci(n):", "Explain why the sky is blue in one sentence."]
PARAMS = SamplingParams(max_new_tokens=24, temperature=0.0, top_p=1.0, stop_on_eos=False)


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


def run(models, **kw):
    tok, t, d = models
    eng = ContinuousBatchingEngine(t, tok, block_size=16, kv_budget_mb=256, max_num_seqs=4,
                                   max_model_len=512, **kw)
    res = eng.generate(PROMPTS, PARAMS)
    run.sources = [r["token_sources"] for r in res]
    return eng, [r["token_ids"] for r in res]


@pytest.mark.skipif(not HAS_CPP_ENGINE, reason="native extension not built")
def test_graphed_draft_is_lossless_and_matches_eager_draft(models):
    _, d = models[0], models[2]
    _, plain = run(models)
    _, eager = run(models, draft_model=d, lookahead_k=4)
    eager_sources = run.sources
    eng, graphed = run(models, draft_model=d, lookahead_k=4, draft_cuda_graph=True)
    assert graphed == plain, "graphed speculation changed the tokens"
    assert graphed == eager
    # Speculation is lossless whatever the draft does, so tokens alone cannot
    # catch a broken draft: it would only lower acceptance. The draft must
    # propose exactly what the eager draft proposes, i.e. identical sources.
    assert run.sources == eager_sources, "graphed draft proposes different tokens"
    assert any("draft" in str(x) for seq in run.sources for x in seq), "nothing was ever accepted"
    # the graph really ran, and every slot came back to the pool
    pool = eng._draft_pool
    assert any(g is not None for g in pool._graph)
    assert len(pool._free) == 4


@pytest.mark.skipif(not HAS_CPP_ENGINE, reason="native extension not built")
def test_slots_are_reused_across_waves_of_requests(models):
    d = models[2]
    eng, _ = run(models, draft_model=d, lookahead_k=3, draft_cuda_graph=True)
    first = [r["token_ids"] for r in eng.generate(PROMPTS, PARAMS)]
    second = [r["token_ids"] for r in eng.generate(PROMPTS, PARAMS)]
    assert first == second and len(eng._draft_pool._free) == 4
