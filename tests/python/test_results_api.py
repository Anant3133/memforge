"""The dashboard's data endpoint must reflect the archived runs, exactly and cheaply."""
import json
import pathlib

from fastapi import FastAPI
from fastapi.testclient import TestClient

from server import results_data


def write(runs, name, fname, payload):
    d = runs / name
    d.mkdir(parents=True, exist_ok=True)
    (d / fname).write_text(json.dumps(payload))


def make_runs(tmp_path):
    runs = tmp_path / "runs"
    write(runs, "phaseA-graphed-r1-M", "context_sweep.json", {
        "platform": {"device": "cuda:0"}, "config": {"model": "org/M"},
        "rows": [{"context_tokens": 1024, "relative_throughput": 0.8, "output_identical": True,
                  "graphed": {"paged_graphed_vs_contiguous_eager": 1.2, "paged_graphed_vs_contiguous_graphed": 0.9,
                              "paged_graphed_output_matches_contiguous": True}}]})
    write(runs, "phaseA-graphed-r2-M", "context_sweep.json", {
        "platform": {"device": "cuda:0"}, "config": {"model": "org/M"},
        "rows": [{"context_tokens": 1024, "relative_throughput": 0.8, "output_identical": True,
                  "graphed": {"paged_graphed_vs_contiguous_eager": 1.4, "paged_graphed_vs_contiguous_graphed": 0.9,
                              "paged_graphed_output_matches_contiguous": True}}]})
    write(runs, "exp-bgraph-r1-p-k3-batched", "speculative.json", {"rows": [
        {"batch_size": 2, "output_identical": False, "output_equivalent": True, "speedup": 0.9,
         "speculative": {"draft_acceptance_rate": 0.5}}]})
    return runs


def test_build_reports_medians_ranges_and_counts(tmp_path):
    d = results_data.build(make_runs(tmp_path))
    row = d["phase_a"][0]
    assert row["model"] == "M" and row["context"] == 1024
    assert row["graphed_vs_stock"] == {"median": 1.3, "min": 1.2, "max": 1.4, "n": 2}
    assert d["equivalence"] == {"identical": 0, "explained_ties": 1, "unexplained": 0}
    assert d["kpis"]["run_directories"] == 3 and d["hard_failures"] == []


def test_endpoint_serves_real_archived_runs():
    from server.api.routes import router
    app = FastAPI()
    app.include_router(router)
    r = TestClient(app).get("/v1/results")
    assert r.status_code == 200
    body = r.json()
    for key in ("kpis", "scoreboard", "phase_a", "speculative", "chunked", "prefix", "qwen14b",
                "equivalence", "stress", "inventory", "tests"):
        assert key in body, key
    assert body["kpis"]["run_directories"] > 100 and body["kpis"]["hard_failures"] == 0
    assert body["equivalence"]["unexplained"] == 0
    assert len(body["stress"]) >= 5 and all(s["leaked_blocks"] == 0 for s in body["stress"])


def test_cache_returns_same_object_until_files_change(tmp_path):
    runs = make_runs(tmp_path)
    results_data._cache.update(stamp=None, payload=None)
    a = results_data.get_results(runs)
    assert results_data.get_results(runs) is a
    write(runs, "exp-bgraph-r2-p-k3-batched", "speculative.json", {"rows": []})
    assert results_data.get_results(runs) is not a
