"""The divergence explainer must separate a near-tie from a genuine bug."""
import torch

from memforge_ml.eval.divergence import (
    explain_divergence, first_divergence, near_tie_tolerance, summarize,
)


class Stub(torch.nn.Module):
    """A 'model' that returns fixed logits at the last position."""

    def __init__(self, logits, dtype=torch.float16):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
        self._l = torch.tensor(logits, dtype=torch.float32)
        self.config = type("C", (), {"num_hidden_layers": 32})()

    def forward(self, input_ids, use_cache=False):
        out = torch.zeros(1, input_ids.shape[1], len(self._l))
        out[0, -1] = self._l
        return type("O", (), {"logits": out})()


def test_identical_runs_report_nothing():
    assert explain_divergence(Stub([1.0, 2.0]), [1], [5, 6], [5, 6], "cpu") is None
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None


def test_near_tie_is_explained():
    m = Stub([10.0, 20.0, 20.01, 0.0])  # tokens 1 and 2 are 0.01 apart
    r = explain_divergence(m, [0], [1], [2], "cpu")
    assert r["explained"] and r["logit_gap"] < r["tolerance"]


def test_genuine_bug_is_not_explained():
    m = Stub([10.0, 20.0, 3.0, 0.0])  # the other run picked a token 17 below the top
    r = explain_divergence(m, [0], [1], [2], "cpu")
    assert not r["explained"] and r["gap_below_top"] > 10


def test_length_mismatch_is_unexplained():
    r = explain_divergence(Stub([1.0, 2.0]), [0], [1, 1], [1], "cpu")
    assert r["kind"] == "length" and not r["explained"]


def test_tolerance_scales_with_dtype():
    f16 = near_tie_tolerance(torch.float16, 20.0)
    bf16 = near_tie_tolerance(torch.bfloat16, 20.0)
    f32 = near_tie_tolerance(torch.float32, 20.0)
    assert f32 < f16 < bf16


def test_summary_equivalent_only_if_every_divergence_explained():
    ok = {"explained": True}
    bad = {"explained": False}
    assert summarize([None, ok])["equivalent"]
    assert not summarize([None, ok, bad])["equivalent"]
