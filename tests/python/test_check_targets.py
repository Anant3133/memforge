"""The regression gate must fail on correctness breaks and regressions, and pass otherwise."""
import importlib.util
import json
import pathlib

spec = importlib.util.spec_from_file_location(
    "check_targets", pathlib.Path(__file__).resolve().parents[2] / "scripts" / "check_targets.py")
ct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ct)


def write(runs, name, fname, payload):
    d = runs / name
    d.mkdir(parents=True, exist_ok=True)
    (d / fname).write_text(json.dumps(payload))


def sweep(identical=True, ratio=1.0):
    return {"rows": [{"context_tokens": 1024, "output_identical": identical,
                      "graphed": {"paged_graphed_vs_contiguous_eager": ratio,
                                  "paged_graphed_output_matches_contiguous": True}}]}


def run_gate(tmp_path, *extra):
    return ct.main(["--runs", str(tmp_path / "runs"), "--baseline", str(tmp_path / "base.json"), *extra])


def test_clean_runs_pass_and_baseline_round_trips(tmp_path):
    write(tmp_path / "runs", "phaseA-graphed-r1-M", "context_sweep.json", sweep(ratio=1.2))
    assert run_gate(tmp_path, "--update") == 0
    assert run_gate(tmp_path) == 0


def test_output_mismatch_is_a_hard_failure(tmp_path):
    write(tmp_path / "runs", "phaseA-graphed-r1-M", "context_sweep.json", sweep(identical=False))
    assert run_gate(tmp_path) == 1


def test_leaked_blocks_are_a_hard_failure(tmp_path):
    write(tmp_path / "runs", "gpu-r1-M", "prefix_sharing.json",
          {"rows": [{"concurrency": 4, "block_ratio": 0.33, "output_identical": True, "blocks_leaked_on": 2}]})
    assert run_gate(tmp_path) == 1


def test_a_drop_beyond_tolerance_is_a_regression_but_a_small_one_is_not(tmp_path):
    runs = tmp_path / "runs"
    write(runs, "phaseA-graphed-r1-M", "context_sweep.json", sweep(ratio=1.20))
    assert run_gate(tmp_path, "--update") == 0
    write(runs, "phaseA-graphed-r1-M", "context_sweep.json", sweep(ratio=1.17))   # -2.5 %
    assert run_gate(tmp_path) == 0
    write(runs, "phaseA-graphed-r1-M", "context_sweep.json", sweep(ratio=1.00))   # -17 %
    assert run_gate(tmp_path) == 1


def test_known_miss_below_target_but_unchanged_does_not_fail(tmp_path):
    write(tmp_path / "runs", "phaseA-graphed-r1-M", "context_sweep.json", sweep(ratio=0.92))
    assert run_gate(tmp_path, "--update") == 0
    assert run_gate(tmp_path) == 0


def test_speculative_rows_are_gated_only_once_classified(tmp_path):
    runs = tmp_path / "runs"
    row = {"batch_size": 2, "output_identical": False, "speedup": 0.8}
    write(runs, "exp-bgraph-r1-p-k3-batched", "speculative.json", {"rows": [row]})
    assert run_gate(tmp_path) == 0                       # pre-explainer: not gated
    row["output_equivalent"] = False
    write(runs, "exp-bgraph-r1-p-k3-batched", "speculative.json", {"rows": [row]})
    assert run_gate(tmp_path) == 1                       # classified and unexplained
