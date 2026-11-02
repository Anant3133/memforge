"""int8 weight-only layers: close to the float layer, correct shapes, replace in place."""
import torch
import torch.nn as nn

from memforge_ml.models.int8_linear import Int8Linear, quantize_linears_int8


def test_matches_float_linear_within_quantisation_error():
    torch.manual_seed(0)
    lin = nn.Linear(256, 64, bias=True)
    q = Int8Linear.from_linear(lin, "cpu")
    x = torch.randn(3, 5, 256)
    ref, got = lin(x), q(x)
    assert got.shape == ref.shape
    rel = (got - ref).norm() / ref.norm()
    assert rel < 0.02, rel            # per-channel int8: ~1% error


def test_exact_when_weights_are_representable():
    w = torch.randint(-127, 128, (8, 16)).float()
    w[:, 0] = 127.0                   # per-row max 127 -> scale exactly 1
    lin = nn.Linear(16, 8, bias=False)
    lin.weight.data = w
    q = Int8Linear.from_linear(lin, "cpu")
    x = torch.randn(4, 16)
    assert torch.allclose(q(x), lin(x), atol=1e-4)


def test_memory_is_about_a_quarter_of_fp32():
    lin = nn.Linear(1024, 1024, bias=False)
    q = Int8Linear.from_linear(lin, "cpu")
    assert q.weight_q.element_size() * q.weight_q.numel() * 4 <= lin.weight.element_size() * lin.weight.numel() * 1.01


def test_quantize_replaces_all_but_skipped():
    m = nn.Sequential()
    m.add_module("a", nn.Linear(8, 8)); m.add_module("lm_head", nn.Linear(8, 8))
    inner = nn.Sequential(nn.Linear(8, 8)); m.add_module("blk", inner)
    n = quantize_linears_int8(m, "cpu")
    assert n == 2 and isinstance(m.a, Int8Linear) and isinstance(m.blk[0], Int8Linear)
    assert isinstance(m.lm_head, nn.Linear)


def test_real_model_top1_agreement():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    try:
        tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
        ref = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, local_files_only=True).eval()
        qm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, local_files_only=True).eval()
    except OSError:
        import pytest; pytest.skip("model not cached")
    quantize_linears_int8(qm, "cpu")
    ids = tok("The capital of France is Paris, and the capital of Italy is", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        a, b = ref(ids).logits[0].argmax(-1), qm(ids).logits[0].argmax(-1)
    assert (a == b).float().mean() >= 0.7
