"""'auto' turns the CUDA-graph paths on only where they can work, and never breaks a CPU engine."""
import warnings

import pytest
import torch

from memforge_ml.runtime.continuous_engine import HAS_CPP_ENGINE, ContinuousBatchingEngine
from memforge_ml.runtime.sampling import SamplingParams

pytestmark = pytest.mark.skipif(not HAS_CPP_ENGINE, reason="native extension not built")
TARGET, DRAFT = "HuggingFaceTB/SmolLM2-360M-Instruct", "HuggingFaceTB/SmolLM2-135M-Instruct"


def load(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(TARGET, local_files_only=True)
        t = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.float32, local_files_only=True).to(device).eval()
        d = AutoModelForCausalLM.from_pretrained(DRAFT, dtype=torch.float32, local_files_only=True).to(device).eval()
    except OSError:
        pytest.skip("models not cached")
    return tok, t, d


def test_auto_is_a_safe_no_op_on_cpu():
    tok, t, d = load("cpu")
    eng = ContinuousBatchingEngine(t, tok, block_size=16, kv_budget_mb=64, max_num_seqs=2, max_model_len=256,
                                   draft_model=d, lookahead_k=2, paged_cuda_graph="auto", draft_cuda_graph="auto")
    assert eng._paged_graph is None and eng._draft_pool is None
    out = eng.generate(["The capital of France is"], SamplingParams(max_new_tokens=8, temperature=0.0, stop_on_eos=False))
    assert len(out[0]["token_ids"]) == 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_auto_enables_graphs_on_cuda_and_matches_eager():
    from memforge_ml.runtime.batched_speculative import BatchedGraphedDraftPool
    tok, t, d = load("cuda")
    prompts = ["The capital of France is", "def fibonacci(n):"]
    params = SamplingParams(max_new_tokens=16, temperature=0.0, top_p=1.0, stop_on_eos=False)

    def run(**kw):
        eng = ContinuousBatchingEngine(t, tok, block_size=16, kv_budget_mb=128, max_num_seqs=2, max_model_len=256, **kw)
        return eng, [r["token_ids"] for r in eng.generate(prompts, params)]

    _, plain = run()
    eng, got = run(draft_model=d, lookahead_k=3, paged_cuda_graph="auto", draft_cuda_graph="auto")
    assert eng._paged_graph is not None and isinstance(eng._draft_pool, BatchedGraphedDraftPool)
    assert got == plain


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_per_sequence_draft_graph_warns_as_deprecated():
    tok, t, d = load("cuda")
    with pytest.warns(DeprecationWarning, match="batched"):
        ContinuousBatchingEngine(t, tok, block_size=16, kv_budget_mb=64, max_num_seqs=2, max_model_len=128,
                                 draft_model=d, lookahead_k=2, draft_cuda_graph=True)
