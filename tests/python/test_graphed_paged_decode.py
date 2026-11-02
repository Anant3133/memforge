"""
Graphed paged decode must change speed only: tokens identical to the eager paged
engine, including prefix-shared prompts (copy-on-write blocks), prompts of very
different lengths in one batch, contexts that cross a gather-bucket boundary,
and more requests than slots.
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

from memforge_ml.runtime.continuous_engine import HAS_CPP_ENGINE, ContinuousBatchingEngine
from memforge_ml.runtime.sampling import SamplingParams

TARGET = "HuggingFaceTB/SmolLM2-360M-Instruct"
SHARED = "The history of the printing press began in the fifteenth century when " * 6
PROMPTS = ["The capital of France is", SHARED + "Gutenberg", SHARED + "Many scholars", "def f(n):",
           "Explain why the sky is blue in one sentence, briefly."]
GREEDY = SamplingParams(max_new_tokens=40, temperature=0.0, top_p=1.0, stop_on_eos=False)
pytestmark = [pytestmark, pytest.mark.skipif(not HAS_CPP_ENGINE, reason="native extension not built")]


@pytest.fixture(scope="module")
def models():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(TARGET, local_files_only=True)
        t = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.float32, local_files_only=True)
    except OSError:
        pytest.skip("model not cached")
    return tok, t.cuda().eval()


def run(models, prompts=PROMPTS, slots=4, **kw):
    tok, t = models
    eng = ContinuousBatchingEngine(t, tok, block_size=16, kv_budget_mb=256, max_num_seqs=slots,
                                   max_model_len=1024, **kw)
    return eng, [r["token_ids"] for r in eng.generate(prompts, GREEDY)]


@pytest.mark.parametrize("slots", [1, 2, 4])
def test_tokens_identical_to_eager_paged(models, slots):
    _, ref = run(models, slots=slots)
    eng, got = run(models, slots=slots, paged_cuda_graph=True)
    assert got == ref
    assert eng._paged_graph.replays > 0, "the graph never ran"


def test_crosses_context_bucket(models):
    tok = models[0]
    unit = "Paged attention stores keys and values in blocks. "
    text = unit
    while len(tok(text)["input_ids"]) < 110:  # 40 new tokens then cross the 128 boundary
        text += unit
    long_prompt = [text]
    _, ref = run(models, long_prompt, slots=1)
    eng, got = run(models, long_prompt, slots=1, paged_cuda_graph=True)
    assert got == ref
    assert eng._paged_graph.captures >= 2, "expected two context buckets to be captured"


def test_pool_matches_eager_pool_after_generation(models):
    """The graph writes KV in-graph; the pool must hold what the eager path wrote."""
    one = [PROMPTS[1]]
    e1, _ = run(models, one, slots=1)
    e2, _ = run(models, one, slots=1, paged_cuda_graph=True)
    n = 64  # compare the leading used region of both pools
    k1, k2 = e1.pool.key_pool.flatten(1, 2)[:, :n].float(), e2.pool.key_pool.flatten(1, 2)[:, :n].float()
    assert torch.allclose(k1, k2, atol=1e-3), (k1 - k2).abs().max()


def test_block_table_longer_than_needed_is_accepted(models):
    """A table with spare trailing blocks must not overflow the graph's table buffer."""
    from memforge_ml.runtime.graphed_paged_decode import GraphedPagedDecoder
    from memforge_ml.runtime.kv_pool import ModelKVSpec, PagedKVPool

    _, model = models
    spec = ModelKVSpec.from_config(model.config, torch.float32, "cuda")
    pool = PagedKVPool(spec, num_blocks=64, block_size=16)
    dec = GraphedPagedDecoder(model, pool, max_batch=1, max_model_len=64, ctx_bucket=16)
    table = torch.arange(60, dtype=torch.long, device="cuda")  # far more than max_blocks
    logits = dec.step([1], [3], [table])
    assert logits.shape[0] == 1 and torch.isfinite(logits).all()
