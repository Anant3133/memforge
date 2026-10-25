#!/usr/bin/env python3
"""
End-to-end benchmark of the live MemForge engine on real model weights.

Four measurements, each against an honest baseline:

  1. Paging overhead   -- MemForge paged KV vs stock Hugging Face contiguous KV,
                          single sequence. Isolates what paging *costs*.
  2. Continuous batching -- N concurrent requests through one engine vs the same
                          N requests served one after another.
  3. Memory efficiency -- how many concurrent sequences fit in a fixed KV budget
                          with block-granular paging vs worst-case contiguous
                          reservation.
  4. Speculative decoding -- acceptance rate, tokens per target forward pass, and
                          wall-clock against plain decoding.

Results are written to benchmarks/results/live_engine.json.

Usage:
    python benchmarks/run_live_engine.py --tokens 32 --concurrency 1 2 4 8
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from memforge_ml.runtime.continuous_engine import ContinuousBatchingEngine  # noqa: E402
from memforge_ml.runtime.kv_pool import ModelKVSpec, PagedKVPool  # noqa: E402
from memforge_ml.runtime.sampling import SamplingParams  # noqa: E402
from memforge_ml.runtime.speculative_runtime import PagedSpeculativeEngine  # noqa: E402

from memforge_ml.model_defaults import (  # noqa: E402
    DEFAULT_DRAFT_MODEL as DEFAULT_DRAFT,
    DEFAULT_TARGET_MODEL as DEFAULT_TARGET,
)

PROMPTS = [
    "Virtual memory lets an operating system",
    "The main advantage of a paged cache is",
    "In three sentences, explain why GPUs are fast at matrix multiplication:",
    "A scheduler that batches requests continuously can",
    "Explain the difference between latency and throughput:",
    "Speculative decoding works by",
    "The key insight behind attention is",
    "Memory fragmentation happens when",
]


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def row(label: str, value: Any) -> None:
    print(f"  {label:<44} {value}")


def _warm_up(engine) -> None:
    """
    Burn one short generation before timing.

    The first forward pass through a freshly built engine pays one-off costs
    (lazy kernel selection, allocator first touch). Timing that would credit
    whichever path happens to run second.
    """
    engine.generate("warm up", SamplingParams(max_new_tokens=2, temperature=0.0, top_p=1.0))


# --------------------------------------------------------------------------- 1

def bench_paging_overhead(
    model, tokenizer, num_tokens: int, repeats: int = 5
) -> Dict[str, Any]:
    """
    Single sequence: MemForge paged KV cache vs stock contiguous KV cache.

    Single-shot timings of this comparison swing by tens of percent on a busy
    CPU, enough to flip the sign of the result, so both paths are measured
    `repeats` times and compared on the median.
    """
    banner("1. Paging overhead (single sequence, greedy)")
    prompt = PROMPTS[0]
    greedy = SamplingParams(max_new_tokens=num_tokens, temperature=0.0, top_p=1.0)
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    engine = ContinuousBatchingEngine(
        model, tokenizer, num_blocks=128, max_num_seqs=4, max_model_len=512
    )

    # Warm both paths before any timing.
    with torch.no_grad():
        model.generate(ids, max_new_tokens=4, do_sample=False,
                       pad_token_id=tokenizer.eos_token_id)
    _warm_up(engine)

    baseline_rates: List[float] = []
    paged_rates: List[float] = []
    identical = True
    last_result: Dict[str, Any] = {}

    for _ in range(repeats):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=num_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        baseline_s = time.perf_counter() - t0
        baseline_rates.append((out.shape[1] - ids.shape[1]) / baseline_s)

        t0 = time.perf_counter()
        last_result = engine.generate(prompt, greedy)[0]
        paged_s = time.perf_counter() - t0
        paged_rates.append(last_result["generated_tokens"] / paged_s)

        identical &= out[0, ids.shape[1]:].tolist() == last_result["token_ids"]

    baseline_tps = statistics.median(baseline_rates)
    paged_tps = statistics.median(paged_rates)

    row("repeats", repeats)
    row("baseline (HF contiguous KV) tok/s, median",
        f"{baseline_tps:.2f}  (min {min(baseline_rates):.2f}, max {max(baseline_rates):.2f})")
    row("MemForge (paged KV) tok/s, median",
        f"{paged_tps:.2f}  (min {min(paged_rates):.2f}, max {max(paged_rates):.2f})")
    row("relative throughput (median)", f"{paged_tps / baseline_tps:.2f}x")
    row("MemForge TTFT (ms)", last_result["time_to_first_token_ms"])
    row("output identical to baseline", identical)
    row("KV pool", f"{engine.pool.describe()['total_mb']} MB / "
                   f"{engine.pool.capacity_tokens} tokens")

    return {
        "tokens": num_tokens,
        "repeats": repeats,
        "baseline_tokens_per_s_median": round(baseline_tps, 3),
        "paged_tokens_per_s_median": round(paged_tps, 3),
        "baseline_tokens_per_s_samples": [round(r, 3) for r in baseline_rates],
        "paged_tokens_per_s_samples": [round(r, 3) for r in paged_rates],
        "relative_throughput": round(paged_tps / baseline_tps, 4),
        "paged_ttft_ms": last_result["time_to_first_token_ms"],
        "output_identical_to_baseline": identical,
        "kv_pool_mb": engine.pool.describe()["total_mb"],
    }


# --------------------------------------------------------------------------- 2

def bench_continuous_batching(
    model, tokenizer, num_tokens: int, concurrency_levels: List[int]
) -> Dict[str, Any]:
    """
    N requests in flight at once vs the same N served sequentially.

    Both paths use the same engine code; the only difference is whether the
    requests are submitted together (so the scheduler can interleave them) or
    one at a time. That isolates the effect of continuous batching.
    """
    banner("2. Continuous batching vs sequential service")
    greedy = SamplingParams(max_new_tokens=num_tokens, temperature=0.0, top_p=1.0)
    results = []

    print(f"  {'N':>3}  {'sequential s':>12}  {'batched s':>10}  "
          f"{'speedup':>8}  {'agg tok/s':>10}  {'tok/fwd':>8}  {'identical':>9}")
    print("  " + "-" * 74)

    for n in concurrency_levels:
        prompts = [PROMPTS[i % len(PROMPTS)] for i in range(n)]

        # Sequential: one engine, but each request submitted and drained before
        # the next one is admitted, so no interleaving can happen. Reusing the
        # engine keeps the KV pool allocated once -- allocating a fresh pool per
        # request would charge the baseline for memory setup instead of compute.
        seq_engine = ContinuousBatchingEngine(
            model, tokenizer, num_blocks=256, max_num_seqs=1, max_model_len=512
        )
        _warm_up(seq_engine)
        seq_outputs = []
        t0 = time.perf_counter()
        for prompt in prompts:
            seq_outputs.append(seq_engine.generate(prompt, greedy)[0])
        sequential_s = time.perf_counter() - t0

        # Continuous: all requests submitted up front into one engine.
        engine = ContinuousBatchingEngine(
            model, tokenizer, num_blocks=256, max_num_seqs=max(n, 1), max_model_len=512
        )
        _warm_up(engine)
        t0 = time.perf_counter()
        batch_outputs = engine.generate(prompts, greedy)
        batched_s = time.perf_counter() - t0

        total_tokens = sum(r["generated_tokens"] for r in batch_outputs)
        identical = all(
            a["token_ids"] == b["token_ids"] for a, b in zip(seq_outputs, batch_outputs)
        )
        entry = {
            "concurrency": n,
            "sequential_s": round(sequential_s, 3),
            "batched_s": round(batched_s, 3),
            "speedup": round(sequential_s / batched_s, 3) if batched_s else None,
            "aggregate_tokens_per_s": round(total_tokens / batched_s, 2),
            "tokens_per_forward_pass": engine.telemetry()["tokens_per_forward_pass"],
            "engine_steps": engine.total_steps,
            "forward_passes": engine.total_forward_passes,
            "outputs_identical_to_sequential": identical,
        }
        results.append(entry)
        print(f"  {n:>3}  {sequential_s:>12.2f}  {batched_s:>10.2f}  "
              f"{entry['speedup']:>7.2f}x  {entry['aggregate_tokens_per_s']:>10.2f}  "
              f"{entry['tokens_per_forward_pass']:>8.2f}  {str(identical):>9}")

    return {"tokens_per_request": num_tokens, "levels": results}


# --------------------------------------------------------------------------- 3

def bench_memory_efficiency(model, num_tokens: int) -> Dict[str, Any]:
    """
    Concurrency that fits in a fixed KV budget: block-granular paging vs the
    contiguous reservation a non-paged engine must make for max_model_len.
    """
    banner("3. KV memory efficiency (paged vs contiguous reservation)")
    spec = ModelKVSpec.from_config(model.config, torch.float32, "cpu")
    budget_mb = 256.0
    block_size = 16
    max_model_len = 1024
    actual_len = 128  # what a typical request really uses

    num_blocks = PagedKVPool.blocks_for_budget(spec, block_size, budget_mb)
    pool_tokens = num_blocks * block_size

    paged_seqs = pool_tokens // actual_len
    contiguous_seqs = pool_tokens // max_model_len
    waste = 1.0 - (actual_len / max_model_len)

    row("KV budget", f"{budget_mb:.0f} MB")
    row("bytes per token (all layers, K+V)", f"{spec.bytes_per_token:,}")
    row("physical blocks in pool", f"{num_blocks:,} x {block_size} tokens")
    row("pool capacity", f"{pool_tokens:,} tokens")
    row("max_model_len (worst case per request)", max_model_len)
    row("actual tokens used per request", actual_len)
    row("concurrent seqs, contiguous reservation", contiguous_seqs)
    row("concurrent seqs, MemForge paging", paged_seqs)
    row("capacity gain", f"{paged_seqs / max(contiguous_seqs, 1):.1f}x")
    row("reservation waste avoided", f"{waste * 100:.1f}%")

    return {
        "kv_budget_mb": budget_mb,
        "bytes_per_token": spec.bytes_per_token,
        "num_blocks": num_blocks,
        "block_size": block_size,
        "pool_capacity_tokens": pool_tokens,
        "max_model_len": max_model_len,
        "actual_tokens_per_request": actual_len,
        "concurrent_seqs_contiguous": contiguous_seqs,
        "concurrent_seqs_paged": paged_seqs,
        "capacity_gain": round(paged_seqs / max(contiguous_seqs, 1), 2),
        "reservation_waste_avoided_pct": round(waste * 100, 2),
    }


# --------------------------------------------------------------------------- 4

def bench_speculative(
    target, tokenizer, draft, num_tokens: int, lookahead_k: int
) -> Dict[str, Any]:
    """Speculative decoding: statistical behaviour and real wall-clock effect."""
    banner("4. Speculative decoding (draft + target verification)")
    engine = PagedSpeculativeEngine(
        target, tokenizer, draft, tokenizer,
        lookahead_k=lookahead_k, num_blocks=128, max_model_len=512,
    )

    per_prompt = []
    for prompt in PROMPTS[:4]:
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        t0 = time.perf_counter()
        with torch.no_grad():
            reference = target.generate(
                ids, max_new_tokens=num_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        baseline_s = time.perf_counter() - t0
        ref_tokens = reference[0, ids.shape[1]:].tolist()

        result = engine.generate(
            prompt, SamplingParams(max_new_tokens=num_tokens, temperature=0.0, top_p=1.0)
        )
        per_prompt.append(
            {
                "prompt": prompt,
                "baseline_s": round(baseline_s, 3),
                "speculative_s": result["total_time_s"],
                "speedup": round(baseline_s / result["total_time_s"], 3),
                "acceptance_rate": result["acceptance_rate"],
                "tokens_per_target_pass": result["tokens_per_target_pass"],
                "target_forward_passes": result["target_forward_passes"],
                "draft_forward_passes": result["draft_forward_passes"],
                "rolled_back_tokens": result["rolled_back_tokens"],
                "lossless_vs_greedy": ref_tokens == result["token_ids"],
            }
        )

    def mean(key: str) -> float:
        return round(sum(p[key] for p in per_prompt) / len(per_prompt), 4)

    all_lossless = all(p["lossless_vs_greedy"] for p in per_prompt)
    row("prompts measured", len(per_prompt))
    row("lookahead K", lookahead_k)
    row("mean acceptance rate", f"{mean('acceptance_rate') * 100:.1f}%")
    row("mean tokens per target forward pass", mean("tokens_per_target_pass"))
    row("mean wall-clock speedup", f"{mean('speedup'):.2f}x")
    row("output identical to greedy target", all_lossless)
    if mean("speedup") < 1.0:
        print()
        print("  NOTE: speculation is a net loss at this scale. It only pays when a")
        print("  target forward pass costs far more than K draft passes; here the")
        print("  target/draft ratio is small and both run on CPU. The statistical")
        print("  guarantee still holds exactly -- see 'output identical to greedy'.")

    return {
        "lookahead_k": lookahead_k,
        "tokens": num_tokens,
        "mean_acceptance_rate": mean("acceptance_rate"),
        "mean_tokens_per_target_pass": mean("tokens_per_target_pass"),
        "mean_speedup": mean("speedup"),
        "all_outputs_lossless": all_lossless,
        "per_prompt": per_prompt,
    }


# --------------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description="MemForge live engine benchmark")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--lookahead-k", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5,
                        help="timed repeats for the paging-overhead comparison")
    parser.add_argument("--target-model", default=DEFAULT_TARGET)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT)
    parser.add_argument("--skip-speculative", action="store_true")
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "benchmarks" / "results" / "live_engine.json"),
    )
    args = parser.parse_args()

    print("MemForge live engine benchmark")
    row("target model", args.target_model)
    row("draft model", "(skipped)" if args.skip_speculative else args.draft_model)
    row("device", "cuda" if torch.cuda.is_available() else "cpu")
    row("torch", torch.__version__)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    target = AutoModelForCausalLM.from_pretrained(args.target_model, dtype=dtype).to(device)
    target.eval()

    report: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": device,
            "dtype": str(dtype),
            "cuda_available": torch.cuda.is_available(),
            "cpu_count": os.cpu_count(),
        },
        "models": {"target": args.target_model, "draft": args.draft_model},
    }

    report["paging_overhead"] = bench_paging_overhead(
        target, tokenizer, args.tokens, repeats=args.repeats
    )
    report["continuous_batching"] = bench_continuous_batching(
        target, tokenizer, args.tokens, args.concurrency
    )
    report["memory_efficiency"] = bench_memory_efficiency(target, args.tokens)

    if not args.skip_speculative:
        draft = AutoModelForCausalLM.from_pretrained(args.draft_model, dtype=dtype).to(device)
        draft.eval()
        report["speculative"] = bench_speculative(
            target, tokenizer, draft, args.tokens, args.lookahead_k
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    banner("Summary")
    paging = report["paging_overhead"]
    row("paged KV vs contiguous KV throughput (median)",
        f"{paging['relative_throughput']:.2f}x")
    row("paged output identical to baseline", paging["output_identical_to_baseline"])
    best = max(report["continuous_batching"]["levels"], key=lambda e: e["speedup"] or 0)
    row(f"best batching speedup (N={best['concurrency']})", f"{best['speedup']:.2f}x")
    row("tokens per forward pass at that N", best["tokens_per_forward_pass"])
    row("paged concurrency gain", f"{report['memory_efficiency']['capacity_gain']}x")
    if "speculative" in report:
        spec = report["speculative"]
        row("speculative acceptance rate", f"{spec['mean_acceptance_rate'] * 100:.1f}%")
        row("speculative tokens/target pass", spec["mean_tokens_per_target_pass"])
        row("speculative wall-clock", f"{spec['mean_speedup']:.2f}x")
        row("speculative output lossless", spec["all_outputs_lossless"])
    print()
    row("results written to", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
