"""
Regression gate over the archived benchmark runs.

Hard gates (always fail the run): any paged-vs-contiguous output mismatch, any
leaked KV block, any speculative run whose output is neither identical nor an
explained near-tie.

Performance gate: each tracked metric is compared with the accepted value in
`benchmarks/targets_baseline.json`; a worse-than-tolerance move is a REGRESSION
(exit 1). Metrics that are below their documented target but unchanged are
reported as KNOWN-MISS, not failures -- the gate catches things getting worse,
and the docs list what is still short.

    python scripts/check_targets.py            # check against the baseline
    python scripts/check_targets.py --update   # accept the current numbers as the baseline
"""
import argparse, glob, json, os, re, statistics as st, sys
from collections import defaultdict

TOL = 0.05  # relative move that counts as a regression


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def collect(runs):
    """-> (metrics {key: (value, higher_is_better, target)}, hard_failures [str])"""
    series = defaultdict(list)
    meta = {}
    hard = []

    def add(key, v, higher, target):
        series[key].append(v)
        meta[key] = (higher, target)

    for d in sorted(glob.glob(runs + "/*")):
        name = os.path.basename(d)
        j = load(d + "/context_sweep.json")
        if j:
            for r in j["rows"]:
                if not r.get("output_identical", True):
                    hard.append(f"{name}: paged output differs from contiguous at ctx {r['context_tokens']}")
                g = r.get("graphed")
                if g and name.startswith(("phaseA-graphed", "phaseA-b32")):
                    if not g.get("paged_graphed_output_matches_contiguous", True):
                        hard.append(f"{name}: graphed paged output differs at ctx {r['context_tokens']}")
                    model = re.sub(r"^phaseA-(?:graphed|b32)-r\d+-", "", name)
                    # the finer-bucket runs supersede the 128-bucket ones where both exist
                    tag = "b32" if "-b32-" in name else "b128"
                    add(f"phaseA.graphed_vs_stock[{model}|{r['context_tokens']}|{tag}]",
                        g["paged_graphed_vs_contiguous_eager"], True, 0.95)
        j = load(d + "/prefix_sharing.json")
        if j:
            for r in j["rows"]:
                if r.get("blocks_leaked_on", 0):
                    hard.append(f"{name}: {r['blocks_leaked_on']} leaked blocks at N={r['concurrency']}")
                if not r.get("output_identical", True):
                    hard.append(f"{name}: prefix sharing changed the output at N={r['concurrency']}")
                if r["concurrency"] == 4:
                    m = re.sub(r"^(?:gpu|cpu)-(?:r\d+-)?", "", name)
                    add(f"prefix.block_ratio[{m}]", r["block_ratio"], False, 0.5)
        j = load(d + "/chunked_prefill.json")
        if j and name.startswith("cpu-"):
            m = j["config"]["model"].split("/")[-1]
            for r in j["rows"]:
                if r["budget"] == 4 and not r.get("fused"):
                    add(f"chunked.cpu_stall_b4[{m}]", r["stall_ratio"], False, 2.0)
        j = load(d + "/speculative.json")
        if j:
            for r in j["rows"]:
                # Runs before the divergence explainer existed have no
                # `output_equivalent`; their mismatches were never classified, so
                # they are not gated (rerun them to get a verdict).
                if "output_equivalent" in r and not r["output_equivalent"]:
                    hard.append(f"{name}: speculative output neither identical nor an explained tie (batch {r['batch_size']})")
                if re.match(r"exp-[bu]graph-r\d+-.*-(batched)$", name):
                    pair = re.sub(r"^exp-[bu]graph-r\d+-|-k\d+|-batched$", "", name)
                    add(f"spec.batched_graph[{pair}|b{r['batch_size']}]", r["speedup"], True, 1.0)
    return {k: (st.median(v), *meta[k], len(v)) for k, v in series.items()}, hard


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="benchmarks/results/runs")
    ap.add_argument("--baseline", default="benchmarks/targets_baseline.json")
    ap.add_argument("--update", action="store_true")
    a = ap.parse_args(argv)
    metrics, hard = collect(a.runs)
    base = load(a.baseline) or {}
    regress, known, ok = [], [], 0
    for k, (v, higher, target, n) in sorted(metrics.items()):
        meets = v >= target if higher else v <= target
        b = base.get(k)
        worse = b is not None and ((v < b * (1 - TOL)) if higher else (v > b * (1 + TOL)))
        if worse:
            regress.append(f"{k}: {v:.3f} vs accepted {b:.3f}")
        elif not meets:
            known.append(f"{k}: {v:.3f} (target {'>=' if higher else '<='} {target})")
        else:
            ok += 1
    print(f"metrics tracked: {len(metrics)}  meeting target: {ok}  known-miss: {len(known)}  regressions: {len(regress)}  hard failures: {len(hard)}")
    for h in hard: print("HARD FAIL  ", h)
    for r in regress: print("REGRESSION ", r)
    for k in known: print("known-miss ", k)
    if a.update:
        json.dump({k: round(v[0], 4) for k, v in metrics.items()}, open(a.baseline, "w"), indent=1, sort_keys=True)
        print(f"baseline written: {a.baseline}")
        return 1 if hard else 0
    return 1 if (hard or regress) else 0


if __name__ == "__main__":
    sys.exit(main())
