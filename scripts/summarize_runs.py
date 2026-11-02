"""Aggregate benchmarks/results/runs/<device>-r<N>-<model>/ into one markdown table, with spread across rounds."""
import glob, json, re, statistics as st, sys
from collections import defaultdict
R = "benchmarks/results/runs/"
ctx, pfx, chk, spec = (defaultdict(list) for _ in range(4))
def load(p):
    try: return json.load(open(p))
    except Exception: return None
for d in sorted(glob.glob(R + "gpu-*") + glob.glob(R + "cpu-*")):
    m = re.match(r".*/(gpu|cpu)-(?:r(\d+)-)?(.+)$", d)
    if not m or "contended" in d or "smoke" in d: continue
    dev, rnd, model = m.groups(); key = (dev, model)
    j = load(d + "/context_sweep.json")
    if j:
        for r in j["rows"]: ctx[key + (r["context_tokens"],)].append((r["relative_throughput"], r["output_identical"], r.get("verdict_decided")))
    j = load(d + "/prefix_sharing.json")
    if j:
        for r in j["rows"]: pfx[key + (r["concurrency"],)].append((r["block_ratio"], r["prefill_ratio"], r["output_identical"], r["blocks_leaked_on"]))
    j = load(d + "/chunked_prefill.json")
    if j:
        for r in j["rows"]: chk[key + (r["budget"], r["fused"])].append(r["stall_ratio"])
    j = load(d + "/speculative.json")
    if j:
        for r in j["rows"]: spec[key + (r["batch_size"],)].append((r["speedup"], r["output_identical"]))
def band(v): return f"{st.median(v):.3f} [{min(v):.3f}-{max(v):.3f}] n={len(v)}"
out = ["# Aggregated results across rounds\n", "Median [min-max] over independent rounds. GPU numbers were taken with a third-party GPU miner running (see report).\n"]
out += ["\n## Phase A: paged / contiguous throughput\n", "| dev | model | ctx | paged/contig | outputs identical |", "|---|---|---:|---|---|"]
for k, v in sorted(ctx.items()): out.append(f"| {k[0]} | {k[1]} | {k[2]} | {band([x[0] for x in v])} | {all(x[1] for x in v)} |")
out += ["\n## Phase B: prefix sharing\n", "| dev | model | N | block ratio | prefill ratio | identical | leaked |", "|---|---|---:|---:|---:|---|---:|"]
for k, v in sorted(pfx.items()): out.append(f"| {k[0]} | {k[1]} | {k[2]} | {st.median(x[0] for x in v):.3f} | {st.median(x[1] for x in v):.3f} | {all(x[2] for x in v)} | {max(x[3] for x in v)} |")
out += ["\n## Phase C: decode stall ratio (lower is better)\n", "| dev | model | budget | fused | stall ratio |", "|---|---|---:|---|---|"]
for k, v in sorted(chk.items()): out.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {band(v)} |")
out += ["\n## Phase D: speculative speedup (>1 = win)\n", "| dev | model | batch | speedup | lossless |", "|---|---|---:|---|---|"]
for k, v in sorted(spec.items()): out.append(f"| {k[0]} | {k[1]} | {k[2]} | {band([x[0] for x in v])} | {all(x[1] for x in v)} |")
open(sys.argv[1] if len(sys.argv) > 1 else R + "AGGREGATE.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
