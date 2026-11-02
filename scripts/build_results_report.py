"""Build docs/RESULTS_REPORT.md (+ runs/REPORT.json) from the archived runs: every number with n and range."""
import glob, json, re, statistics as st
from collections import defaultdict
R = "benchmarks/results/runs/"
def load(p):
    try: return json.load(open(p))
    except Exception: return None
def band(v, f="{:.3f}"):
    v = [x for x in v if x is not None]
    if not v: return "n/a"
    return f"{f.format(st.median(v))} [{f.format(min(v))}-{f.format(max(v))}] n={len(v)}"
report, out = {}, ["# MemForge results report (A100 server)\n",
  "Every figure is the median over independent runs with [min-max] and n. GPU timings were taken on a card shared with other tenants (100% utilisation from jobs that are not ours), so ratios between arms run back to back are meaningful; absolute tokens/s are not.\n"]

# 1. batched draft graph
spec = defaultdict(list); eq = defaultdict(list); acc = defaultdict(list)
for d in sorted(glob.glob(R + "exp-bgraph-r*") + glob.glob(R + "exp-ugraph-r*")):
    m = re.match(r".*/exp-[bu]graph-r(\d+)-(.+)-(eager|perseq|batched|unrolled)$", d)
    j = load(d + "/speculative.json")
    if not (m and j): continue
    _, pair, arm = m.groups()
    for r in j["rows"]:
        k = (pair, r["batch_size"], arm)
        spec[k].append(r["speedup"]); eq[k].append(r.get("output_equivalent", r["output_identical"]))
        acc[k].append(r["speculative"]["draft_acceptance_rate"])
out += ["\n## 1. Batched CUDA-graph draft (speculative speedup vs plain decoding, K=3, fp16)\n",
        "Speedup > 1 means speculation beats plain decoding. Arms are run back to back in the same round.\n",
        "| pair | batch | eager draft | per-sequence graph | batched graph | **batched, one replay/round** | acceptance | output equivalent |", "|---|---:|---|---|---|---|---|---|"]
report["batched_draft"] = {}
for pair in sorted({k[0] for k in spec}):
    for b in sorted({k[1] for k in spec if k[0] == pair}):
        cell = lambda a: band(spec.get((pair, b, a), []))
        e = all(eq.get((pair, b, "batched"), [False]))
        a_ = acc.get((pair, b, "batched"), [])
        e = e and all(eq.get((pair, b, "unrolled"), [True]))
        out.append(f"| {pair} | {b} | {cell('eager')} | {cell('perseq')} | {cell('batched')} | **{cell('unrolled')}** | {st.median(a_):.3f} | {e} |" if a_ else f"| {pair} | {b} | {cell('eager')} | {cell('perseq')} | {cell('batched')} | **{cell('unrolled')}** | n/a | {e} |")
        report["batched_draft"][f"{pair}|b{b}"] = {a: spec.get((pair, b, a), []) for a in ("eager", "perseq", "batched", "unrolled")}

# 2. phase A graphed
ga = defaultdict(lambda: defaultdict(list))
for d in sorted(glob.glob(R + "phaseA-graphed-r*")):
    m = re.match(r".*/phaseA-graphed-r(\d+)-(.+)$", d); j = load(d + "/context_sweep.json")
    if not (m and j): continue
    for r in j["rows"]:
        g = r.get("graphed")
        if not g: continue
        k = (m.group(2), r["context_tokens"])
        ga[k]["eager"].append(r["relative_throughput"])
        ga[k]["pg_vs_ce"].append(g["paged_graphed_vs_contiguous_eager"])
        ga[k]["pg_vs_cg"].append(g["paged_graphed_vs_contiguous_graphed"])
        ga[k]["match"].append(g["paged_graphed_output_matches_contiguous"])
out += ["\n## 2. Phase A with the paged decode step as a CUDA graph (fp16)\n",
        "Target: paged within 5% of contiguous (>= 0.95). Three baselines are shown because the answer depends on which one you mean.\n",
        "| model | ctx | eager paged / eager contiguous (before) | **graphed paged / eager contiguous** | graphed paged / graphed contiguous | output = eager contiguous |", "|---|---:|---|---|---|---|"]
report["phaseA_graphed"] = {}
for (m, c), v in sorted(ga.items()):
    out.append(f"| {m} | {c} | {band(v['eager'])} | **{band(v['pg_vs_ce'])}** | {band(v['pg_vs_cg'])} | {all(v['match'])} |")
    report["phaseA_graphed"][f"{m}|{c}"] = dict(v)

# 3. CPU chunked prefill repeats
ck = defaultdict(list)
for d in sorted(glob.glob(R + "cpu-*")):
    j = load(d + "/chunked_prefill.json")
    if not j: continue
    model = j["config"]["model"].split("/")[-1]
    for r in j["rows"]:
        if not r.get("fused"): ck[(model, r["budget"])].append(r["stall_ratio"])
out += ["\n## 3. Chunked prefill on CPU, repeated (worst decode stall / normal step; target <= 2.0)\n",
        "| model | unchunked | budget 64 | budget 16 | **budget 4** | meets target at budget 4 in every run |", "|---|---|---|---|---|---|"]
report["cpu_chunked"] = {}
for m in sorted({k[0] for k in ck}):
    g = lambda b: ck.get((m, b), [])
    out.append(f"| {m} | {band(g(0), '{:.2f}')} | {band(g(64), '{:.2f}')} | {band(g(16), '{:.2f}')} | **{band(g(4), '{:.2f}')}** | {all(x <= 2.0 for x in g(4)) if g(4) else 'n/a'} |")
    report["cpu_chunked"][m] = {b: g(b) for b in (0, 64, 16, 4)}

open("docs/RESULTS_REPORT.md", "w").write("\n".join(out) + "\n")
json.dump(report, open(R + "REPORT.json", "w"), indent=1)
print("\n".join(out))
