#!/usr/bin/env python3
"""
Phase B benchmark: prefix sharing via copy-on-write, on real model weights.

Answers three questions, each against the same engine with sharing switched off
so the comparison is like for like:

  1. **Memory.** How many physical KV blocks do N requests sharing one long
     system prompt actually occupy? This is the Phase B definition of done.
  2. **Compute.** A cache hit means the prefix's KV is already resident, so
     prefill skips it. How many prefill tokens and forward passes does that
     save, and does it show up in time to first token?
  3. **Correctness.** Is the output byte-identical to the non-sharing engine?
     This is the one that matters: every failure mode of prefix sharing produces
     fluent, confident, *wrong* text rather than a crash, so the benchmark
     asserts exact token equality rather than reporting a similarity score.

Results are written to benchmarks/results/prefix_sharing.json.

Usage:
    python benchmarks/run_prefix_sharing.py
    python benchmarks/run_prefix_sharing.py --concurrency 1 2 4 8 --tokens 16
    python benchmarks/run_prefix_sharing.py --prefix-tokens 512
"""

from __future__ import annotations

import argparse
import json
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
from memforge_ml.runtime.sampling import SamplingParams  # noqa: E402

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
RESULTS_PATH = PROJECT_ROOT / "benchmarks" / "results" / "prefix_sharing.json"


# --------------------------------------------------------------------- helpers

def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}")


def build_prompts(tokenizer, num_requests: int, prefix_tokens: int) -> List[List[int]]:
    """
    `num_requests` prompts sharing one long prefix, each with its own short tail.

    Built as token ids rather than text so the shared prefix is exactly the
    requested length -- the whole measurement turns on how many full 16-token
    blocks it covers, and tokenizing prose would make that approximate.
    """
    source = (
        "You are MemForge, a careful and precise assistant. Answer accurately, "
        "cite your reasoning, prefer concrete detail over generalities, and say "
        "plainly when you do not know something. "
    ) * 64
    ids = tokenizer(source, return_tensors=None)["input_ids"]
    if len(ids) < prefix_tokens:
        raise RuntimeError(
            f"system prompt source is only {len(ids)} tokens, need {prefix_tokens}"
        )
    prefix = ids[:prefix_tokens]

    # Distinct tails, so the sequences genuinely diverge after the shared prefix.
    tails = [
        tokenizer(f" Question {i}: explain briefly.", return_tensors=None)["input_ids"]
        for i in range(num_requests)
    ]
    return [prefix + tail for tail in tails]


def run_batch(model, tokenizer, prompts, *, sharing: bool, max_new_tokens: int,
              num_blocks: int, block_size: int) -> Dict[str, Any]:
    """One engine, one batch of concurrent requests, with sharing on or off."""
    engine = ContinuousBatchingEngine(
        model,
        tokenizer,
        block_size=block_size,
        num_blocks=num_blocks,
        max_num_seqs=max(8, len(prompts)),
        max_model_len=max(len(p) for p in prompts) + max_new_tokens + 8,
        enable_prefix_sharing=sharing,
    )
    params = SamplingParams(max_new_tokens=max_new_tokens, temperature=0.0, stop_on_eos=False)

    seq_ids = [engine.submit(p, params) for p in prompts]
    peak_blocks = 0
    # Sampled inside the loop: by the time the batch finishes, every sequence has
    # released its references, so reading these at the end reports 0 no matter
    # how much sharing happened.
    peak_shared = 0
    peak_saved = 0
    started = time.perf_counter()
    while engine.has_work():
        engine.step()
        peak_blocks = max(peak_blocks, engine.allocator.get_num_allocated_blocks())
        peak_shared = max(peak_shared, engine.allocator.get_num_shared_blocks())
        peak_saved = max(peak_saved, engine.allocator.get_num_blocks_saved())
    wall = time.perf_counter() - started
    engine.step()  # let the scheduler reap the finished sequences

    results = [engine.result_for(i) for i in seq_ids]
    telemetry = engine.telemetry()
    ttfts = [r["time_to_first_token_ms"] for r in results if r["time_to_first_token_ms"]]

    return {
        "sharing": sharing,
        "requests": len(prompts),
        "peak_blocks": peak_blocks,
        "peak_kv_mb": round(peak_blocks * engine.pool.bytes_per_block / (1024 * 1024), 2),
        "shared_blocks_peak": peak_shared,
        "blocks_saved": peak_saved,
        "memory_saved_mb": round(peak_saved * engine.pool.bytes_per_block / (1024 * 1024), 2),
        "prefix_hit_rate": telemetry["prefix_hit_rate"],
        "prefill_tokens": engine.total_prefill_tokens,
        "cached_prefix_tokens": engine.total_cached_prefix_tokens,
        "forward_passes": engine.total_forward_passes,
        "scheduler_steps": engine.total_steps,
        "cow_copies": engine.total_cow_copies,
        "wall_s": round(wall, 3),
        "median_ttft_ms": round(statistics.median(ttfts), 2) if ttfts else None,
        "tokens_generated": sum(r["generated_tokens"] for r in results),
        "token_ids": [r["token_ids"] for r in results],
        "blocks_after_teardown": engine.allocator.get_num_allocated_blocks(),
        "free_after_teardown": engine.allocator.get_num_free_blocks(),
        "total_blocks": engine.allocator.get_total_blocks(),
        "registry_after_teardown": (
            engine.prefix_cache.num_cached_blocks() if engine.prefix_cache else 0
        ),
    }


# ----------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description="MemForge prefix sharing benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--tokens", type=int, default=16, help="tokens generated per request")
    parser.add_argument("--prefix-tokens", type=int, default=512,
                        help="length of the shared system prompt, in tokens")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=600)
    args = parser.parse_args()

    banner("MemForge Phase B -- Prefix Sharing (Copy-on-Write)")
    print(f"model            : {args.model}")
    print(f"shared prefix    : {args.prefix_tokens} tokens")
    print(f"generated tokens : {args.tokens} per request")
    print(f"pool             : {args.num_blocks} blocks x {args.block_size} tokens")
    print(f"device           : cpu (torch {torch.__version__})")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    rows: List[Dict[str, Any]] = []
    mismatches: List[int] = []

    for n in args.concurrency:
        prompts = build_prompts(tokenizer, n, args.prefix_tokens)
        banner(f"{n} concurrent request(s), {len(prompts[0])}-token prompts")

        off = run_batch(model, tokenizer, prompts, sharing=False, max_new_tokens=args.tokens,
                        num_blocks=args.num_blocks, block_size=args.block_size)
        on = run_batch(model, tokenizer, prompts, sharing=True, max_new_tokens=args.tokens,
                       num_blocks=args.num_blocks, block_size=args.block_size)

        identical = off["token_ids"] == on["token_ids"]
        if not identical:
            mismatches.append(n)

        ratio = on["peak_blocks"] / off["peak_blocks"] if off["peak_blocks"] else 0.0
        prefill_ratio = (
            on["prefill_tokens"] / off["prefill_tokens"] if off["prefill_tokens"] else 0.0
        )

        print(f"  {'':24}{'OFF':>12}{'ON':>12}")
        print(f"  {'peak blocks':24}{off['peak_blocks']:>12}{on['peak_blocks']:>12}")
        print(f"  {'peak KV MB':24}{off['peak_kv_mb']:>12}{on['peak_kv_mb']:>12}")
        print(f"  {'prefill tokens':24}{off['prefill_tokens']:>12}{on['prefill_tokens']:>12}")
        print(f"  {'forward passes':24}{off['forward_passes']:>12}{on['forward_passes']:>12}")
        print(f"  {'median TTFT ms':24}{str(off['median_ttft_ms']):>12}{str(on['median_ttft_ms']):>12}")
        print(f"  {'wall seconds':24}{off['wall_s']:>12}{on['wall_s']:>12}")
        print(f"  {'shared blocks':24}{off['shared_blocks_peak']:>12}{on['shared_blocks_peak']:>12}")
        print(f"  {'prefix hit rate':24}{off['prefix_hit_rate']:>12}{on['prefix_hit_rate']:>12}")
        print(f"\n  block ratio ON/OFF   : {ratio:.3f}")
        print(f"  prefill ratio ON/OFF : {prefill_ratio:.3f}")
        print(f"  output byte-identical: {'YES' if identical else 'NO -- INVESTIGATE'}")
        print(f"  cow copies           : {on['cow_copies']}")

        leak_off = off["total_blocks"] - off["free_after_teardown"] - off["blocks_after_teardown"]
        leak_on = on["total_blocks"] - on["free_after_teardown"] - on["blocks_after_teardown"]
        print(f"  blocks leaked        : OFF {leak_off}, ON {leak_on}")

        rows.append({
            "concurrency": n,
            "prompt_tokens": len(prompts[0]),
            "block_ratio": round(ratio, 4),
            "prefill_ratio": round(prefill_ratio, 4),
            "output_identical": identical,
            "blocks_leaked_off": leak_off,
            "blocks_leaked_on": leak_on,
            # token_ids are dropped from the saved record: they are only needed
            # for the equality check and would dominate the file.
            "off": {k: v for k, v in off.items() if k != "token_ids"},
            "on": {k: v for k, v in on.items() if k != "token_ids"},
        })

    banner("Summary")
    print(f"{'N':>4}{'blocks OFF':>13}{'blocks ON':>12}{'ratio':>9}{'prefill ratio':>15}{'identical':>12}")
    for row in rows:
        print(
            f"{row['concurrency']:>4}{row['off']['peak_blocks']:>13}"
            f"{row['on']['peak_blocks']:>12}{row['block_ratio']:>9.3f}"
            f"{row['prefill_ratio']:>15.3f}{str(row['output_identical']):>12}"
        )

    best = min(rows, key=lambda r: r["block_ratio"])
    print(f"\nBest memory reduction: {1 / best['block_ratio']:.1f}x at N={best['concurrency']} "
          f"({best['off']['peak_blocks']} -> {best['on']['peak_blocks']} blocks)")

    if mismatches:
        print(f"\nFAILED: output differed from the non-sharing engine at N={mismatches}. "
              "Prefix sharing must be byte-identical; treat this as a correctness bug.")
    else:
        print("\nOutput byte-identical to the non-sharing engine at every concurrency level.")

    payload = {
        "benchmark": "prefix_sharing",
        "phase": "B",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": {
            "system": platform.system(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "config": {
            "model": args.model,
            "prefix_tokens": args.prefix_tokens,
            "generated_tokens": args.tokens,
            "block_size": args.block_size,
            "num_blocks": args.num_blocks,
            "concurrency": args.concurrency,
        },
        "all_outputs_identical": not mismatches,
        "rows": rows,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH.relative_to(PROJECT_ROOT)}")

    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
