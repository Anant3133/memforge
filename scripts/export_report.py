#!/usr/bin/env python3
"""
Turn benchmark JSON into Markdown tables ready to paste into a report.

Reads whatever is in `benchmarks/results/` and renders each known schema into
the table shape that the progress document already uses. Markdown first, because
that is what actually gets pasted into documents; `--format csv` is there for
spreadsheets and `--charts` for PNGs.

Each results file has its own shape -- `prefix_sharing.json` is a list of
off/on pairs, `live_engine.json` is grouped by scenario, `stress_test.json` is
one run with a sample series -- so each gets its own renderer rather than a
generic dict-to-table walk, which would produce something unreadable.

Robustness matters more than coverage here: a missing file means "not run yet",
and an unrecognised schema is skipped with a note rather than taking the whole
report down. Someone generating a report an hour before a presentation should
get the sections that exist.

Usage:
    python scripts/export_report.py
    python scripts/export_report.py --out benchmarks/results/REPORT.md
    python scripts/export_report.py --only prefix_sharing --format csv
    python scripts/export_report.py --charts
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"


# ----------------------------------------------------------------- utilities

def table(headers: List[str], rows: List[List[Any]], aligns: Optional[List[str]] = None) -> str:
    """A GitHub-flavoured Markdown table. Numeric columns are right-aligned."""
    if aligns is None:
        aligns = ["---"] * len(headers)
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "| " + " | ".join(aligns) + " |"]
    for row in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in row) + " |")
    return "\n".join(out)


def right(n: int) -> List[str]:
    return ["---:"] * n


def fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "**no**"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def ratio_arrow(before: Any, after: Any) -> str:
    """`a -> b` with the ratio, the shape the progress doc already uses."""
    if before in (None, 0) or after is None:
        return "--"
    return f"{before} -> {after} ({after / before:.3f}x)"


# ----------------------------------------------------------------- renderers

def render_prefix_sharing(data: Dict[str, Any]) -> Tuple[str, List[Tuple[str, List[List[Any]]]]]:
    config = data.get("config", {})
    rows: List[List[Any]] = []
    csv_rows: List[List[Any]] = []

    for row in data.get("rows", []):
        off, on = row.get("off", {}), row.get("on", {})
        rows.append([
            row.get("concurrency"),
            f"{off.get('peak_blocks')} -> {on.get('peak_blocks')}",
            fmt(row.get("block_ratio"), 3),
            f"{off.get('peak_kv_mb')} -> {on.get('peak_kv_mb')}",
            f"{off.get('prefill_tokens')} -> {on.get('prefill_tokens')}",
            f"{fmt(off.get('median_ttft_ms'), 0)} -> {fmt(on.get('median_ttft_ms'), 0)}",
            f"{off.get('wall_s')} -> {on.get('wall_s')}",
            fmt(row.get("output_identical")),
        ])
        csv_rows.append([
            row.get("concurrency"), off.get("peak_blocks"), on.get("peak_blocks"),
            row.get("block_ratio"), off.get("peak_kv_mb"), on.get("peak_kv_mb"),
            off.get("prefill_tokens"), on.get("prefill_tokens"),
            off.get("median_ttft_ms"), on.get("median_ttft_ms"),
            off.get("wall_s"), on.get("wall_s"), row.get("output_identical"),
        ])

    body = [
        "### Prefix sharing (Phase B)",
        "",
        f"Model `{config.get('model', '?')}` · {config.get('prefix_tokens', '?')}-token "
        f"shared prefix · {config.get('generated_tokens', '?')} tokens generated · "
        f"block size {config.get('block_size', '?')}",
        "",
        table(
            ["Concurrency", "Peak blocks", "Ratio", "Peak KV MB", "Prefill tokens",
             "Median TTFT", "Wall s", "Byte-identical"],
            rows, right(8),
        ),
    ]

    if data.get("all_outputs_identical"):
        body += ["", "Output was byte-identical to the non-sharing engine at every "
                     "concurrency level."]
    else:
        body += ["", "**Output differed from the non-sharing engine. Treat as a "
                     "correctness bug, not a tuning issue.**"]

    best = min((r for r in data.get("rows", []) if r.get("block_ratio")),
               key=lambda r: r["block_ratio"], default=None)
    if best:
        body += ["", f"Best memory reduction: **{1 / best['block_ratio']:.1f}x** at "
                     f"N={best['concurrency']}."]

    csv_header = ["concurrency", "blocks_off", "blocks_on", "block_ratio", "kv_mb_off",
                  "kv_mb_on", "prefill_off", "prefill_on", "ttft_off_ms", "ttft_on_ms",
                  "wall_off_s", "wall_on_s", "identical"]
    return "\n".join(body), [("prefix_sharing", [csv_header] + csv_rows)]


def render_stress_test(data: Dict[str, Any]) -> Tuple[str, List[Tuple[str, List[List[Any]]]]]:
    config = data.get("config", {})
    passed = data.get("passed")

    headline = table(
        ["Measurement", "Value"],
        [
            ["Requests", f"{data.get('succeeded')} ok / {data.get('total_requests')} sent"],
            ["Status codes", f"`{data.get('status_counts')}`"],
            ["Duration", fmt(data.get("duration_s"), 1, " s")],
            ["**Blocks leaked**", f"**{data.get('leaked_blocks')}**"],
            ["Blocks allocated, start -> end",
             f"{data.get('blocks_allocated_at_start')} -> {data.get('blocks_allocated_at_end')}"],
            ["Peak blocks allocated", data.get("peak_blocks_allocated")],
            ["Peak blocks shared", data.get("peak_blocks_shared")],
            ["Peak requests in flight", data.get("peak_in_flight")],
            ["RSS initial", fmt(data.get("initial_rss_mb"), 1, " MB")],
            ["RSS plateau (post-warm-up)", fmt(data.get("plateau_rss_mb"), 1, " MB")],
            ["RSS peak", fmt(data.get("peak_rss_mb"), 1, " MB")],
            ["RSS final", fmt(data.get("final_rss_mb"), 1, " MB")],
            ["RSS growth vs plateau", fmt(data.get("rss_growth_after_warmup_mb"), 1, " MB")],
            ["TTFT p50 / p95 / p99",
             f"{fmt(data.get('p50_ttft_ms'), 1)} / {fmt(data.get('p95_ttft_ms'), 1)} / "
             f"{fmt(data.get('p99_ttft_ms'), 1)} ms"],
            ["Mean tokens/sec per request", fmt(data.get("mean_tok_per_sec"), 2)],
            ["Total tokens generated", data.get("total_tokens")],
        ],
    )

    body = [
        "### Sustained load & memory leak (Phase F)",
        "",
        f"{config.get('requests', '?')} requests · concurrency "
        f"{config.get('concurrent_clients', '?')} · {config.get('tokens', '?')} tokens each"
        + (" · SSE streaming" if config.get("stream") else ""),
        "",
        headline,
    ]

    if data.get("verdicts"):
        body += ["", "**Verdicts**", ""]
        body += [f"- {v}" for v in data["verdicts"]]

    if passed is not None:
        body += ["", f"Overall: {'**PASS**' if passed else '**FAIL**'}"]

    body += [
        "",
        "> Block accounting is the exact measurement here: `blocks_leaked` is "
        "`total - free - allocated` and has no warm-up behaviour, so a nonzero value "
        "is a genuine leak. RSS is indicative only -- PyTorch and the allocator both "
        "cache, so it rises over the first requests and then plateaus, which is why "
        "growth is measured against the post-warm-up plateau rather than the first "
        "sample.",
    ]

    samples = data.get("samples", [])
    csv_rows = [["t", "rss_mb", "blocks_allocated", "blocks_free", "blocks_leaked",
                 "blocks_shared", "in_flight"]]
    for s in samples:
        csv_rows.append([s.get("t"), s.get("rss_mb"), s.get("blocks_allocated"),
                         s.get("blocks_free"), s.get("blocks_leaked"),
                         s.get("blocks_shared"), s.get("in_flight")])
    return "\n".join(body), [("stress_test_samples", csv_rows)]


def render_live_engine(data: Dict[str, Any]) -> Tuple[str, List[Tuple[str, List[List[Any]]]]]:
    """
    The original end-to-end benchmark.

    Its shape predates the others and groups results by scenario, so the known
    sections are pulled out by name and anything unrecognised is listed rather
    than guessed at.
    """
    body = ["### Live engine end-to-end"]
    csv_out: List[Tuple[str, List[List[Any]]]] = []

    models = data.get("models") or {}
    env = data.get("environment") or {}
    if models or env:
        body += ["", f"Target `{models.get('target', '?')}` · draft "
                     f"`{models.get('draft', '?')}` · {env.get('device', '?')} · "
                     f"{env.get('dtype', '?')} · torch {env.get('torch', '?')}"]

    # Continuous batching: the headline result of the earlier phase.
    levels = ((data.get("continuous_batching") or {}).get("levels")) or []
    if levels:
        rows, csv_rows = [], [["concurrency", "sequential_s", "batched_s", "speedup",
                               "aggregate_tok_per_s", "tokens_per_forward_pass",
                               "outputs_identical"]]
        for entry in levels:
            rows.append([
                entry.get("concurrency"),
                fmt(entry.get("sequential_s"), 2),
                fmt(entry.get("batched_s"), 2),
                fmt(entry.get("speedup"), 2, "x"),
                fmt(entry.get("aggregate_tokens_per_s"), 1),
                fmt(entry.get("tokens_per_forward_pass"), 2),
                fmt(entry.get("outputs_identical_to_sequential")),
            ])
            csv_rows.append([entry.get("concurrency"), entry.get("sequential_s"),
                             entry.get("batched_s"), entry.get("speedup"),
                             entry.get("aggregate_tokens_per_s"),
                             entry.get("tokens_per_forward_pass"),
                             entry.get("outputs_identical_to_sequential")])
        body += ["", "**Continuous batching**", "",
                 table(["Concurrency", "Sequential s", "Batched s", "Speedup",
                        "Aggregate tok/s", "Tokens/forward pass", "Identical output"],
                       rows, right(7))]
        csv_out.append(("live_engine_batching", csv_rows))

    overhead = data.get("paging_overhead") or {}
    if overhead:
        body += ["", "**Paging overhead** (median of "
                     f"{overhead.get('repeats', '?')} interleaved runs)", "",
                 table(["Path", "tok/s"],
                       [["Stock contiguous KV",
                         fmt(overhead.get("baseline_tokens_per_s_median"), 2)],
                        ["MemForge paged KV",
                         fmt(overhead.get("paged_tokens_per_s_median"), 2)],
                        ["**Relative**",
                         f"**{fmt(overhead.get('relative_throughput'), 2, 'x')}**"]],
                       right(2))]

    memory = data.get("memory_efficiency") or {}
    if memory:
        body += ["", "**Memory efficiency** — concurrent sequences that fit in "
                     f"{memory.get('kv_budget_mb', '?')} MB of KV", "",
                 table(["Allocation strategy", "Concurrent sequences"],
                       [[f"Contiguous reservation for max_model_len="
                         f"{memory.get('max_model_len', '?')}",
                         memory.get("concurrent_seqs_contiguous")],
                        [f"Block-granular paging, "
                         f"{memory.get('actual_tokens_per_request', '?')} tokens used",
                         f"**{memory.get('concurrent_seqs_paged')}**"],
                        ["**Capacity gain**",
                         f"**{fmt(memory.get('capacity_gain'), 2, 'x')}**"]],
                       right(2))]

    spec = data.get("speculative") or {}
    if spec:
        speedup = spec.get("mean_speedup")
        body += ["", f"**Speculative decoding** (lookahead K="
                     f"{spec.get('lookahead_k', '?')})", "",
                 table(["Metric", "Value"],
                       [["Mean draft acceptance rate",
                         fmt((spec.get("mean_acceptance_rate") or 0) * 100, 1, "%")],
                        ["Mean tokens per target forward pass",
                         fmt(spec.get("mean_tokens_per_target_pass"), 2)],
                        ["Output lossless vs greedy target",
                         fmt(spec.get("all_outputs_lossless"))],
                        ["**Mean wall-clock speedup**",
                         f"**{fmt(speedup, 2, 'x')}**"]],
                       right(2))]
        if isinstance(speedup, (int, float)) and speedup < 1:
            # Reported, not buried. This is the project's one negative result
            # and the docs explain it rather than omitting it.
            body += ["", f"> Speculative decoding is **{speedup:.2f}x, i.e. slower** on this "
                         "CPU/model pair. The mechanism is verified lossless and confirms "
                         f"{fmt(spec.get('mean_tokens_per_target_pass'), 2)} tokens per target "
                         "pass, but the 360M/135M cost ratio is too small for that to pay off "
                         "in wall-clock terms on CPU. See PROGRESS_AND_NEXT_PHASES.md §3.5."]

    if len(body) <= 2:
        body += ["", "*No recognised sections; the file may use an older schema.*"]
    return "\n".join(body), csv_out


RENDERERS: Dict[str, Callable[[Dict[str, Any]], Tuple[str, List[Tuple[str, List[List[Any]]]]]]] = {
    "prefix_sharing": render_prefix_sharing,
    "stress_test": render_stress_test,
    "live_engine": render_live_engine,
}


# --------------------------------------------------------------------- charts

def write_charts(name: str, data: Dict[str, Any], out_dir: Path) -> List[Path]:
    """PNG charts, where a picture adds something a table does not."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # no display on a build machine
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib is not installed; skipping charts", file=sys.stderr)
        return []

    written: List[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    if name == "prefix_sharing" and data.get("rows"):
        rows = data["rows"]
        xs = [r["concurrency"] for r in rows]
        off = [r["off"]["peak_blocks"] for r in rows]
        on = [r["on"]["peak_blocks"] for r in rows]

        fig, ax = plt.subplots(figsize=(7, 4))
        width = 0.35
        positions = range(len(xs))
        ax.bar([p - width / 2 for p in positions], off, width, label="sharing off")
        ax.bar([p + width / 2 for p in positions], on, width, label="sharing on")
        ax.set_xticks(list(positions))
        ax.set_xticklabels([str(x) for x in xs])
        ax.set_xlabel("concurrent requests")
        ax.set_ylabel("peak physical blocks")
        ax.set_title("Prefix sharing: peak KV blocks")
        ax.set_ylim(bottom=0)  # bar charts must start at zero
        ax.legend()
        fig.tight_layout()
        path = out_dir / "prefix_sharing_blocks.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)

    if name == "stress_test" and data.get("samples"):
        samples = data["samples"]
        times = [s["t"] - samples[0]["t"] for s in samples]
        rss = [s.get("rss_mb") for s in samples]
        allocated = [s.get("blocks_allocated") for s in samples]

        fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
        if any(v is not None for v in rss):
            top.plot(times, rss, linewidth=1.4)
            top.set_ylabel("RSS (MB)")
            top.set_title("Sustained load: memory and block accounting")
            top.set_ylim(bottom=0)
        bottom.plot(times, allocated, linewidth=1.4, color="tab:green")
        bottom.set_ylabel("blocks allocated")
        bottom.set_xlabel("seconds")
        bottom.set_ylim(bottom=0)
        fig.tight_layout()
        path = out_dir / "stress_test_memory.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)

    return written


# ---------------------------------------------------------------------- main

def load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"  skipping {path.name}: not valid JSON ({exc})", file=sys.stderr)
    except OSError as exc:
        print(f"  skipping {path.name}: {exc}", file=sys.stderr)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Export MemForge benchmark results")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--out", help="write Markdown here instead of stdout")
    parser.add_argument("--only", action="append",
                        help="limit to one result set (repeatable), e.g. --only prefix_sharing")
    parser.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    parser.add_argument("--charts", action="store_true", help="also write PNG charts")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"no results directory at {results_dir}", file=sys.stderr)
        print("run a benchmark first, e.g. python benchmarks/run_prefix_sharing.py",
              file=sys.stderr)
        return 1

    files = sorted(results_dir.glob("*.json"))
    if not files:
        print(f"no JSON results in {results_dir} yet", file=sys.stderr)
        print("run a benchmark first, e.g. python benchmarks/run_stress_test.py",
              file=sys.stderr)
        return 1

    sections: List[str] = []
    csv_sets: List[Tuple[str, List[List[Any]]]] = []
    rendered, skipped, missing = [], [], []

    for path in files:
        name = path.stem
        if args.only and name not in args.only:
            continue
        data = load(path)
        if data is None:
            skipped.append(name)
            continue
        renderer = RENDERERS.get(name)
        if renderer is None:
            # Unknown schema: note it rather than guessing at a layout.
            skipped.append(name)
            continue
        try:
            markdown, csvs = renderer(data)
        except Exception as exc:
            print(f"  {name}: renderer failed ({type(exc).__name__}: {exc})", file=sys.stderr)
            skipped.append(name)
            continue
        sections.append(markdown)
        csv_sets.extend(csvs)
        rendered.append(name)

        if args.charts:
            for chart in write_charts(name, data, results_dir / "charts"):
                print(f"  wrote {chart.relative_to(PROJECT_ROOT)}", file=sys.stderr)

    for expected in RENDERERS:
        if expected not in rendered and not (args.only and expected not in args.only):
            if not (results_dir / f"{expected}.json").exists():
                missing.append(expected)

    if args.format == "csv":
        if not csv_sets:
            print("nothing to export as CSV", file=sys.stderr)
            return 1
        for name, rows in csv_sets:
            buffer = io.StringIO()
            csv.writer(buffer, lineterminator="\n").writerows(rows)
            if args.out:
                path = Path(args.out).with_name(f"{Path(args.out).stem}_{name}.csv")
                path.write_text(buffer.getvalue(), encoding="utf-8")
                print(f"wrote {path}")
            else:
                print(f"# {name}")
                print(buffer.getvalue())
        return 0

    header = [
        "# MemForge Benchmark Report",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by `scripts/export_report.py`.",
        "",
    ]
    if missing:
        header += [
            "> Not yet run: " + ", ".join(f"`{m}`" for m in missing)
            + ". Those sections are absent rather than empty.",
            "",
        ]
    if skipped:
        header += [
            "> Skipped (unreadable or unrecognised schema): "
            + ", ".join(f"`{s}`" for s in skipped) + ".",
            "",
        ]

    document = "\n".join(header) + "\n\n---\n\n".join(sections) + "\n"

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(document, encoding="utf-8")
        print(f"wrote {out_path} ({len(sections)} section"
              f"{'' if len(sections) == 1 else 's'})")
    else:
        sys.stdout.write(document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
