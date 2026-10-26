#!/usr/bin/env python3
"""
Phase A benchmark: paged vs contiguous KV throughput across context lengths.

The Phase A definition of done is a measurement — "paged throughput within 5%
of contiguous at 2,048 tokens" — so this script exists before the optimisations
it is meant to judge, and its first job is to record an honest baseline.

Why a sweep rather than a single number: the existing comparison in
`run_live_engine.py` is taken at ~40 tokens of context, where the paged gather
is cheap. The gather is `O(context)` per layer per step, so the interesting
question is how the ratio degrades as context grows — which a single point
cannot answer.

Three things are measured at each context length:

  1. **Decode throughput**, paged vs stock contiguous `DynamicCache`, single
     sequence. Runs are interleaved and medians compared, because single-shot
     timings on a busy CPU swing far enough to flip the sign of this comparison.
  2. **Batched decode**, which is where limitation 2 lives: the engine gathers
     every sequence, left-pads and concatenates per layer, copying the whole
     batch context every step.
  3. **Output equality.** Paged and contiguous must produce identical token ids.
     Every optimisation in this area changes how KV is read, and a wrong read
     produces fluent, confident, wrong text rather than a crash.

Usage:
    python benchmarks/run_context_sweep.py
    python benchmarks/run_context_sweep.py --contexts 128 512 1024 2048 --steps 8
    python benchmarks/run_context_sweep.py --skip-batched --repeats 5
"""

from __future__ import annotations

import argparse
import json
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
from transformers.cache_utils import DynamicCache  # noqa: E402

from memforge_ml.model_defaults import DEFAULT_TARGET_MODEL  # noqa: E402
from _device import (  # noqa: E402
    add_device_args,
    describe_device,
    peak_memory_mb,
    resolve_device,
)
from memforge_ml.runtime.continuous_engine import ContinuousBatchingEngine  # noqa: E402
from memforge_ml.runtime.kv_pool import ModelKVSpec, PagedKVPool  # noqa: E402
from memforge_ml.runtime.decode_view_cache import capacity_for  # noqa: E402
from memforge_ml.runtime.paged_cache import PagedCache  # noqa: E402
from memforge_ml.runtime.sampling import SamplingParams  # noqa: E402

RESULTS_PATH = PROJECT_ROOT / "benchmarks" / "results" / "context_sweep.json"


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def make_prompt_ids(tokenizer, num_tokens: int) -> List[int]:
    """A prompt of exactly `num_tokens` tokens."""
    source = (
        "Paged attention stores the key and value tensors of a sequence in "
        "fixed size blocks rather than one contiguous buffer, so memory can be "
        "allocated on demand as generation proceeds. "
    ) * 200
    ids = tokenizer(source, return_tensors=None)["input_ids"]
    if len(ids) < num_tokens:
        raise RuntimeError(f"source text yields {len(ids)} tokens, need {num_tokens}")
    return ids[:num_tokens]


def _model_device(model) -> torch.device:
    """
    Where this model's weights live.

    Read from the model rather than passed in: an input tensor built on the
    wrong device is a hard error on GPU ("index is on cpu, different from other
    tensors on cuda:0"), and taking it from the weights makes that impossible
    to get wrong at a call site.
    """
    return next(model.parameters()).device


# ------------------------------------------------------- single-sequence paths

@torch.no_grad()
def decode_contiguous(model, prompt_ids: List[int], steps: int) -> Dict[str, Any]:
    """Stock Hugging Face `DynamicCache`: the baseline paging is judged against."""
    cache = DynamicCache()
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=_model_device(model))

    t_prefill = time.perf_counter()
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    prefill_s = time.perf_counter() - t_prefill

    tokens: List[int] = []
    t0 = time.perf_counter()
    for _ in range(steps):
        token = int(out.logits[0, -1].argmax())
        tokens.append(token)
        out = model(
            input_ids=torch.tensor(
                [[token]], dtype=torch.long, device=_model_device(model)
            ),
            past_key_values=cache,
            use_cache=True,
        )
    decode_s = time.perf_counter() - t0

    return {
        "tokens": tokens,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "tok_per_s": steps / decode_s if decode_s > 0 else 0.0,
    }


@torch.no_grad()
def decode_paged(model, spec: ModelKVSpec, prompt_ids: List[int], steps: int,
                 block_size: int, view_cache: bool = False) -> Dict[str, Any]:
    """
    The same decode loop through `PagedCache`.

    Deliberately bypasses the scheduler: this isolates the *cache read path*,
    which is what Phase A changes. Scheduler cost is measured by
    run_live_engine.py instead.
    """
    total = len(prompt_ids) + steps + 1
    num_blocks = (total + block_size - 1) // block_size + 1
    pool = PagedKVPool(spec, num_blocks=num_blocks, block_size=block_size)
    cache = PagedCache(
        pool,
        list(range(num_blocks)),
        view_capacity=capacity_for(len(prompt_ids), steps) if view_cache else None,
    )

    ids = torch.tensor([prompt_ids], dtype=torch.long, device=_model_device(model))
    t_prefill = time.perf_counter()
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    prefill_s = time.perf_counter() - t_prefill

    tokens: List[int] = []
    t0 = time.perf_counter()
    for _ in range(steps):
        token = int(out.logits[0, -1].argmax())
        tokens.append(token)
        cache.begin_step()
        out = model(
            input_ids=torch.tensor(
                [[token]], dtype=torch.long, device=_model_device(model)
            ),
            past_key_values=cache,
            use_cache=True,
        )
    decode_s = time.perf_counter() - t0

    return {
        "tokens": tokens,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "tok_per_s": steps / decode_s if decode_s > 0 else 0.0,
        "view_cache_mb": (
            cache.view_cache.describe()["mb"] if cache.view_cache else 0.0
        ),
    }


# --------------------------------------------------------------- batched path

def decode_batched(model, tokenizer, prompt_ids: List[int], concurrency: int,
                   steps: int, block_size: int, kv_budget_mb: float) -> Dict[str, Any]:
    """
    Batched decode through the real engine, which is where limitation 2 bites.

    Reports tokens per forward pass alongside throughput: at a fixed context,
    the pad-and-concatenate cost scales with batch x context, so throughput
    alone hides whether batching is paying for itself.
    """
    engine = ContinuousBatchingEngine(
        model,
        tokenizer,
        block_size=block_size,
        kv_budget_mb=kv_budget_mb,
        max_num_seqs=max(8, concurrency),
        max_model_len=len(prompt_ids) + steps + 8,
        enable_prefix_sharing=False,  # isolate the read path from sharing
    )
    params = SamplingParams(max_new_tokens=steps, temperature=0.0, stop_on_eos=False)
    # Distinct suffixes so the batch is genuinely N different sequences.
    for i in range(concurrency):
        engine.submit(prompt_ids + [(i * 97) % 1000], params)

    t0 = time.perf_counter()
    while engine.has_work():
        engine.step()
    wall = time.perf_counter() - t0

    telemetry = engine.telemetry()
    return {
        "wall_s": round(wall, 3),
        "tokens": telemetry["tokens_generated"],
        "tok_per_s": round(telemetry["tokens_generated"] / wall, 2) if wall else 0.0,
        "tokens_per_forward_pass": telemetry["tokens_per_forward_pass"],
        "forward_passes": telemetry["forward_passes"],
    }


# ---------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description="MemForge Phase A context sweep")
    parser.add_argument("--model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--contexts", type=int, nargs="+", default=[128, 512, 1024, 2048])
    parser.add_argument("--steps", type=int, default=8, help="decode steps per run")
    parser.add_argument("--repeats", type=int, default=3,
                        help="interleaved repeats; the median is reported")
    add_device_args(parser)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--kv-budget-mb", type=float, default=1024.0)
    parser.add_argument("--skip-batched", action="store_true")
    parser.add_argument("--view-cache", action="store_true",
                        help="also measure the append-only decode mirror (tier 2)")
    args = parser.parse_args()
    # Resolved here, before any banner or model load, so everything
    # below can rely on it. An explicit --device cuda fails here
    # rather than silently producing CPU numbers labelled cuda.
    device, dtype = resolve_device(args)

    banner("MemForge Phase A — paged vs contiguous across context lengths")
    print(f"model       : {args.model}")
    print(f"contexts    : {args.contexts}")
    print(f"decode steps: {args.steps} per run, {args.repeats} interleaved repeats")
    print(f"block size  : {args.block_size}")
    print(f"device      : {device} {dtype} (torch {torch.__version__})")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()
    # Must match where the model actually is, and in the same dtype. A pool on
    # the wrong device fails deep inside the scatter with "source is on cuda:0,
    # different from other tensors on cpu" -- the hardcoded (float32, cpu) here
    # was invisible until a GPU run existed to expose it.
    spec = ModelKVSpec.from_config(model.config, dtype, device)
    print(f"geometry    : {spec.num_layers} layers, {spec.num_kv_heads} kv heads, "
          f"head_dim {spec.head_dim}, {spec.bytes_per_token} bytes/token")

    rows: List[Dict[str, Any]] = []
    mismatches: List[int] = []

    for ctx in args.contexts:
        prompt = make_prompt_ids(tokenizer, ctx)
        banner(f"context {ctx} tokens")

        # Interleaved, so a CPU frequency change or another process affects both
        # paths rather than only one of them.
        cont_rates, paged_rates = [], []
        cont_tokens = paged_tokens = None
        cont_prefill, paged_prefill = [], []
        view_rates: List[float] = []
        view_tokens = None
        view_mb = 0.0
        for r in range(args.repeats):
            c = decode_contiguous(model, prompt, args.steps)
            p = decode_paged(model, spec, prompt, args.steps, args.block_size)
            cont_rates.append(c["tok_per_s"]); paged_rates.append(p["tok_per_s"])
            cont_prefill.append(c["prefill_s"]); paged_prefill.append(p["prefill_s"])
            cont_tokens, paged_tokens = c["tokens"], p["tokens"]
            line = (f"  repeat {r + 1}: contiguous {c['tok_per_s']:6.2f} tok/s | "
                    f"paged {p['tok_per_s']:6.2f} tok/s")
            if args.view_cache:
                # Interleaved with the other two so all three see the same
                # machine conditions.
                vw = decode_paged(model, spec, prompt, args.steps, args.block_size,
                                  view_cache=True)
                view_rates.append(vw["tok_per_s"]); view_tokens = vw["tokens"]
                view_mb = vw["view_cache_mb"]
                line += f" | paged+mirror {vw['tok_per_s']:6.2f} tok/s"
            print(line)

        cont = statistics.median(cont_rates)
        paged = statistics.median(paged_rates)
        ratio = paged / cont if cont else 0.0
        identical = cont_tokens == paged_tokens
        if not identical:
            mismatches.append(ctx)

        print(f"\n  median contiguous : {cont:.2f} tok/s")
        print(f"  median paged      : {paged:.2f} tok/s")
        print(f"  relative          : {ratio:.3f}x   "
              f"({'within' if abs(1 - ratio) <= 0.05 else 'outside'} 5% of contiguous)")
        print(f"  prefill median    : contiguous {statistics.median(cont_prefill):.3f}s | "
              f"paged {statistics.median(paged_prefill):.3f}s")
        print(f"  output identical  : {'YES' if identical else 'NO -- INVESTIGATE'}")
        if args.view_cache and view_rates:
            view = statistics.median(view_rates)
            if view_tokens != cont_tokens:
                mismatches.append(ctx)
            print(f"  paged + mirror    : {view:.2f} tok/s "
                  f"({view / paged:.3f}x vs paged, {view / cont:.3f}x vs contiguous), "
                  f"costing {view_mb} MB")

        row: Dict[str, Any] = {
            "context_tokens": ctx,
            "contiguous_tok_per_s": round(cont, 3),
            "paged_tok_per_s": round(paged, 3),
            "relative_throughput": round(ratio, 4),
            "within_5_percent": bool(abs(1 - ratio) <= 0.05),
            "contiguous_prefill_s": round(statistics.median(cont_prefill), 4),
            "paged_prefill_s": round(statistics.median(paged_prefill), 4),
            "output_identical": identical,
            "contiguous_samples": [round(x, 3) for x in cont_rates],
            "paged_samples": [round(x, 3) for x in paged_rates],
        }
        if args.view_cache and view_rates:
            row["view_cache"] = {
                "tok_per_s": round(statistics.median(view_rates), 3),
                "vs_paged": round(statistics.median(view_rates) / paged, 4),
                "vs_contiguous": round(statistics.median(view_rates) / cont, 4),
                "mirror_mb": view_mb,
                "output_identical": view_tokens == cont_tokens,
                "samples": [round(x, 3) for x in view_rates],
            }

        if not args.skip_batched:
            batched = decode_batched(model, tokenizer, prompt, args.concurrency,
                                     args.steps, args.block_size, args.kv_budget_mb)
            print(f"  batched (N={args.concurrency}): {batched['tok_per_s']:.2f} tok/s, "
                  f"{batched['tokens_per_forward_pass']:.2f} tok/forward, "
                  f"{batched['wall_s']}s")
            row["batched"] = {"concurrency": args.concurrency, **batched}

        rows.append(row)

    # ------------------------------------------------------------- summary
    banner("Summary")
    print(f"{'context':>9}{'contiguous':>13}{'paged':>11}{'relative':>11}"
          f"{'within 5%':>11}{'identical':>11}")
    for row in rows:
        print(f"{row['context_tokens']:>9}{row['contiguous_tok_per_s']:>13.2f}"
              f"{row['paged_tok_per_s']:>11.2f}{row['relative_throughput']:>11.3f}"
              f"{str(row['within_5_percent']):>11}{str(row['output_identical']):>11}")

    worst = min(rows, key=lambda r: r["relative_throughput"])
    print(f"\nWorst case: {worst['relative_throughput']:.3f}x at "
          f"{worst['context_tokens']} tokens.")

    target = next((r for r in rows if r["context_tokens"] == 2048), None)
    if target:
        gap = (1 - target["relative_throughput"]) * 100
        met = target["within_5_percent"]
        print(f"Phase A target (within 5% at 2048): "
              f"{'MET' if met else f'NOT MET -- {gap:.1f}% slower than contiguous'}")

    if mismatches:
        print(f"\nFAILED: paged output differed from contiguous at {mismatches}. "
              "Treat as a correctness bug, not a performance result.")
    else:
        print("\nPaged output byte-identical to contiguous at every context length.")

    payload = {
        "benchmark": "context_sweep",
        "phase": "A",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": describe_device(device, dtype),
        # None on CPU, not 0.0: torch has no CPU allocator to query, and a zero
        # would read as "measured, and it was nothing".
        "peak_memory_mb": peak_memory_mb(device),
        "config": {
            "model": args.model,
            "contexts": args.contexts,
            "steps": args.steps,
            "repeats": args.repeats,
            "block_size": args.block_size,
            "concurrency": None if args.skip_batched else args.concurrency,
        },
        "model_geometry": {
            "num_layers": spec.num_layers,
            "num_kv_heads": spec.num_kv_heads,
            "head_dim": spec.head_dim,
            "bytes_per_token": spec.bytes_per_token,
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
