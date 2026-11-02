"""
Aggregates the archived benchmark runs into one JSON document for the dashboard.

Everything here is read from `benchmarks/results/runs/<label>/*.json` (the same
files `scripts/build_results_report.py` and `scripts/check_targets.py` use), so
the dashboard cannot show a number that is not in an archived run. Each figure
is a median with its min-max range and the number of runs behind it, because a
single run proves nothing on a GPU shared with other jobs.

The result is cached against the newest file mtime, so polling is cheap and a
new benchmark run shows up on the next request without restarting the server.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import re
import statistics as st
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "benchmarks" / "results" / "runs"
TESTS_STATUS = ROOT / "benchmarks" / "results" / "tests_status.json"

_cache: Dict[str, Any] = {"stamp": None, "payload": None}


def _load(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def band(values: List[float]) -> Optional[Dict[str, Any]]:
    v = [float(x) for x in values if x is not None]
    if not v:
        return None
    return {"median": round(st.median(v), 4), "min": round(min(v), 4), "max": round(max(v), 4), "n": len(v)}


def _classify(label: str) -> Dict[str, Any]:
    """Group, device and round from a run directory name."""
    rnd = re.search(r"-r(\d+)", label)
    groups = [
        (r"^gpu-", "GPU ladder", "gpu"), (r"^cpu-", "CPU ladder", "cpu"),
        (r"^phaseA-", "Phase A graphed paged decode", "gpu"),
        (r"^exp-bgraph|^exp-ugraph", "Batched draft graph", "gpu"),
        (r"^exp-graph", "Per-sequence draft graph", "gpu"),
        (r"^exp-fused", "Fused prefill", "gpu"), (r"^exp-draft", "Draft-size experiments", "gpu"),
        (r"^exp-exact|^exp-lookahead", "Exactness / lookahead", "gpu"),
        (r"^exp-decode", "Decode latency probe", "gpu"), (r"^stress", "Stress / soak", "gpu"),
    ]
    for pat, name, dev in groups:
        if re.search(pat, label):
            return {"group": name, "device": dev, "round": int(rnd.group(1)) if rnd else None}
    return {"group": "Other", "device": "n/a", "round": int(rnd.group(1)) if rnd else None}


def _headline(d: str) -> str:
    j = _load(d + "/context_sweep.json")
    if j and j.get("rows"):
        r = j["rows"][-1]
        return f"paged/contiguous {r['relative_throughput']:.2f}x @ {r['context_tokens']} tok"
    j = _load(d + "/speculative.json")
    if j and j.get("rows"):
        r = j["rows"][-1]
        return f"speculative {r.get('speedup')}x @ batch {r['batch_size']}"
    j = _load(d + "/chunked_prefill.json")
    if j and j.get("rows"):
        r = [x for x in j["rows"] if x["budget"] == 4 and not x.get("fused")]
        if r:
            return f"decode stall {r[0]['stall_ratio']:.2f}x @ budget 4"
    j = _load(d + "/prefix_sharing.json")
    if j and j.get("rows"):
        r = j["rows"][-1]
        return f"prefix blocks {r['block_ratio']:.2f}x @ N={r['concurrency']}"
    j = _load(d + "/stress_test.json")
    if j:
        return f"{j['succeeded']}/{j['total_requests']} ok, leaked {j['leaked_blocks']}"
    return ""


def _parse_stress_log(path: str) -> Optional[dict]:
    """Older soak runs only kept the console log; read the same fields out of it."""
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
        ok = re.search(r"requests\s+(\d+) ok / (\d+) sent", text)
        leaked = re.search(r"blocks leaked \(must be 0\)\s+(\d+)", text)
        rss = re.search(r"RSS growth vs plateau\s+(-?[\d.]+) MB", text)
        ttft = re.search(r"TTFT p50 / p95 / p99\s+[\d.]+ / ([\d.]+) /", text)
        inflight = re.search(r"peak requests in flight\s+(\d+)", text)
        if not (ok and leaked and rss):
            return None
        model = re.search(r"stress-(.+)/stress\.log", path)
        return {"total_requests": int(ok.group(2)), "succeeded": int(ok.group(1)),
                "leaked_blocks": int(leaked.group(1)), "rss_growth_after_warmup_mb": float(rss.group(1)),
                "p95_ttft_ms": float(ttft.group(1)) if ttft else None,
                "peak_in_flight": int(inflight.group(1)) if inflight else None,
                "passed": "FAIL " not in text and int(ok.group(1)) == int(ok.group(2)) and int(leaked.group(1)) == 0,
                "config": {"model": model.group(1) if model else ""}}
    except Exception:
        return None


def _check_targets():
    spec = importlib.util.spec_from_file_location("check_targets", ROOT / "scripts" / "check_targets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _model_of(j: dict) -> str:
    return (j.get("config", {}).get("model") or "").split("/")[-1]


def build(runs_dir: Path = RUNS) -> Dict[str, Any]:
    runs_dir = Path(runs_dir)
    dirs = sorted(p for p in glob.glob(str(runs_dir / "*")) if os.path.isdir(p))
    inventory, models, json_files, rows_total = [], set(), 0, 0
    for d in dirs:
        label = os.path.basename(d)
        files = sorted(os.path.basename(f) for f in glob.glob(d + "/*.json"))
        json_files += len(files)
        for f in files:
            j = _load(os.path.join(d, f))
            if j:
                rows_total += len(j.get("rows", [])) if isinstance(j.get("rows"), list) else 1
                m = _model_of(j)
                if m:
                    models.add(m)
        inventory.append({"label": label, **_classify(label), "files": files,
                          "headline": _headline(d), "mtime": int(os.path.getmtime(d))})

    # ----- Phase A: eager (gpu-r*) vs graphed
    eager = defaultdict(list)
    graphed = defaultdict(lambda: defaultdict(list))
    for d in dirs:
        label = os.path.basename(d)
        j = _load(d + "/context_sweep.json")
        if not j or j.get("platform", {}).get("device") not in ("cuda:0", "cuda"):
            continue
        model = _model_of(j)
        for r in j["rows"]:
            if re.match(r"^gpu-r\d+-", label) or label.startswith("gpu-"):
                eager[(model, r["context_tokens"])].append(r["relative_throughput"])
            g = r.get("graphed")
            if g and label.startswith(("phaseA-graphed", "phaseA-b32")):
                bucket = 32 if "-b32-" in label else 128
                k = (model, r["context_tokens"], bucket)
                graphed[k]["vs_stock"].append(g["paged_graphed_vs_contiguous_eager"])
                graphed[k]["vs_graphed"].append(g["paged_graphed_vs_contiguous_graphed"])
                graphed[k]["match"].append(bool(g["paged_graphed_output_matches_contiguous"]))
    phase_a = []
    for (model, ctx, bucket), v in sorted(graphed.items()):
        phase_a.append({"model": model, "context": ctx, "bucket": bucket,
                        "eager_paged": band(eager.get((model, ctx), [])),
                        "graphed_vs_stock": band(v["vs_stock"]), "graphed_vs_graphed": band(v["vs_graphed"]),
                        "output_identical": all(v["match"])})

    # ----- Speculative
    spec = defaultdict(lambda: defaultdict(list))
    acc = defaultdict(list)
    eq_rows = [0, 0, 0]  # identical, explained, unexplained
    for d in dirs:
        label = os.path.basename(d)
        m = re.match(r"^exp-[bu]graph-r\d+-(.+)-(eager|perseq|batched|unrolled)$", label)
        j = _load(d + "/speculative.json")
        if not j:
            continue
        for r in j["rows"]:
            if "output_equivalent" in r:
                if r["output_identical"]:
                    eq_rows[0] += 1
                elif r["output_equivalent"]:
                    eq_rows[1] += 1
                else:
                    eq_rows[2] += 1
        if m:
            pair = m.group(1).replace("-k3", "")
            for r in j["rows"]:
                spec[(pair, r["batch_size"])][m.group(2)].append(r["speedup"])
                if m.group(2) == "batched":
                    acc[(pair, r["batch_size"])].append(r["speculative"]["draft_acceptance_rate"])
    speculative = [{"pair": p, "batch": b, "acceptance": band(acc.get((p, b), [])),
                    **{arm: band(vals) for arm, vals in arms.items()}} for (p, b), arms in sorted(spec.items())]

    # ----- Chunked prefill, prefix sharing
    chunk = defaultdict(lambda: defaultdict(list))
    prefix = defaultdict(list)
    for d in dirs:
        label = os.path.basename(d)
        if not label.startswith(("gpu-", "cpu-")):
            continue
        j = _load(d + "/chunked_prefill.json")
        backend = "GPU" if label.startswith("gpu-") else "CPU"
        if "int8" in label:
            backend = "GPU int8"
        if j:
            for r in j["rows"]:
                if not r.get("fused"):
                    chunk[(backend, _model_of(j))][r["budget"]].append(r["stall_ratio"])
        j = _load(d + "/prefix_sharing.json")
        if j:
            for r in j["rows"]:
                if r["concurrency"] == 4:
                    prefix[(backend, _model_of(j))].append(
                        (r["block_ratio"], r["prefill_ratio"], r["blocks_leaked_on"], r["output_identical"]))
    chunked = [{"backend": b, "model": m, "budgets": {str(k): band(v) for k, v in sorted(bud.items())}}
               for (b, m), bud in sorted(chunk.items())]
    prefix_out = [{"backend": b, "model": m, "block_ratio": band([x[0] for x in v]),
                   "prefill_ratio": band([x[1] for x in v]), "leaked": max(x[2] for x in v),
                   "identical": all(x[3] for x in v)} for (b, m), v in sorted(prefix.items())]

    # ----- Qwen3-14B
    q14 = {"gpu_int8": {"context": {}, "chunked": {}}, "cpu_fp32": {"context": {}, "chunked": {}}}
    for key, pat in (("gpu_int8", "gpu-*qwen14b-int8"), ("cpu_fp32", "cpu-qwen14b")):
        ctx = defaultdict(list)
        ck = defaultdict(list)
        ident = []
        for d in glob.glob(str(runs_dir / pat)):
            j = _load(d + "/context_sweep.json")
            if j:
                for r in j["rows"]:
                    ctx[r["context_tokens"]].append(r["relative_throughput"])
                    ident.append(r["output_identical"])
            j = _load(d + "/chunked_prefill.json")
            if j:
                for r in j["rows"]:
                    if not r.get("fused"):
                        ck[r["budget"]].append(r["stall_ratio"])
        q14[key] = {"context": {str(k): band(v) for k, v in sorted(ctx.items())},
                    "chunked": {str(k): band(v) for k, v in sorted(ck.items())},
                    "output_identical": all(ident) if ident else None}

    # ----- Stress
    stress = []
    for d in dirs:
        j = _load(d + "/stress_test.json")
        if not j and os.path.exists(d + "/stress.log"):
            j = _parse_stress_log(d + "/stress.log")
        if j:
            stress.append({"label": os.path.basename(d), "model": _model_of(j) or os.path.basename(d), "passed": j["passed"],
                           "requests": j["total_requests"], "succeeded": j["succeeded"],
                           "leaked_blocks": j["leaked_blocks"], "rss_growth_mb": j["rss_growth_after_warmup_mb"],
                           "p95_ttft_ms": j["p95_ttft_ms"], "peak_in_flight": j["peak_in_flight"]})

    # ----- Scoreboard (reuses the regression gate's own metric extraction)
    ct = _check_targets()
    metrics, hard = ct.collect(str(runs_dir))
    baseline = _load(str(ROOT / "benchmarks" / "targets_baseline.json")) or {}
    scoreboard = []
    for k, (v, higher, target, n) in sorted(metrics.items()):
        meets = v >= target if higher else v <= target
        scoreboard.append({"id": k, "section": k.split(".")[0], "value": round(v, 4), "target": target,
                           "higher_is_better": higher, "meets": bool(meets), "n": n,
                           "baseline": baseline.get(k)})

    tests = _load(str(TESTS_STATUS)) or {}
    met = sum(1 for s in scoreboard if s["meets"])
    kpis = {
        "run_directories": len(dirs), "json_files": json_files, "data_rows": rows_total,
        "models": len(models), "targets_tracked": len(scoreboard), "targets_met": met,
        "hard_failures": len(hard),
        "cpu_tests_passed": tests.get("cpu_suite", {}).get("passed"),
        "gpu_tests_passed": tests.get("gpu_suite", {}).get("passed"),
        "best_paged_speedup": max((p["graphed_vs_stock"]["median"] for p in phase_a if p["graphed_vs_stock"]), default=None),
        "fp16_mismatches_explained": eq_rows[1], "fp16_mismatches_unexplained": eq_rows[2],
    }
    return {
        "generated_at": time.time(), "runs_dir": str(runs_dir.relative_to(ROOT)) if runs_dir.is_relative_to(ROOT) else str(runs_dir),
        "kpis": kpis, "models": sorted(models), "scoreboard": scoreboard, "hard_failures": hard,
        "phase_a": phase_a, "speculative": speculative, "chunked": chunked, "prefix": prefix_out,
        "qwen14b": q14, "equivalence": {"identical": eq_rows[0], "explained_ties": eq_rows[1], "unexplained": eq_rows[2]},
        "stress": stress, "tests": tests, "inventory": inventory,
    }


def get_results(runs_dir: Path = RUNS) -> Dict[str, Any]:
    """Cached `build()`; recomputed when any run file changes."""
    files = glob.glob(str(Path(runs_dir) / "*" / "*.json")) + [str(TESTS_STATUS)]
    stamp = (len(files), max((os.path.getmtime(f) for f in files if os.path.exists(f)), default=0), str(runs_dir))
    if _cache["stamp"] != stamp:
        _cache["payload"], _cache["stamp"] = build(runs_dir), stamp
    return _cache["payload"]
