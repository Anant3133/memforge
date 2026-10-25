#!/usr/bin/env python3
"""
Phase C benchmark: does a long prompt arriving stall requests already in flight?

The Phase C definition of done is a latency claim, not a throughput one:
*"p99 inter-token latency for an in-flight request is unaffected by a 2,000-token
prompt arriving"*. So this measures exactly that, and nothing else.

The experiment:

  1. Submit a short request and step until it is decoding steadily.
  2. Mid-flight, submit a long prompt.
  3. Keep stepping, recording the wall-clock gap between each of the *first*
     request's tokens.
  4. Compare that gap distribution with chunked prefill off and on.

With chunking off, the long prompt is prefilled in a single forward pass, and
the victim produces no token during that step — one enormous gap in its stream.
With chunking on, the prompt is processed a budget at a time while decoding
continues every step, so the gap distribution should barely move.

Reported as the ratio of worst-case gap to the victim's own median gap, because
the absolute numbers depend entirely on machine and model and the *shape* is the
claim. A ratio near 1 means decoding never noticed; a large ratio is a stall.

Usage:
    python benchmarks/run_chunked_prefill.py
    python benchmarks/run_chunked_prefill.py --prompt-tokens 2048 --budgets 0 512 256 128
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

from memforge_ml.model_defaults import DEFAULT_TARGET_MODEL  # noqa: E402
from memforge_ml.runtime.continuous_engine import ContinuousBatchingEngine  # noqa: E402
from memforge_ml.runtime.sampling import SamplingParams  # noqa: E402

RESULTS_PATH = PROJECT_ROOT / "benchmarks" / "results" / "chunked_prefill.json"


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def build_prompt(tokenizer, num_tokens: int) -> List[int]:
    source = (
        "Chunked prefill splits a long prompt across scheduler steps so that one "
        "large request cannot monopolise a forward pass and stall the decode of "
        "every request already in flight. "
    ) * 200
    ids = tokenizer(source, return_tensors=None)["input_ids"]
    if len(ids) < num_tokens:
        raise RuntimeError(f"source yields {len(ids)} tokens, need {num_tokens}")
    return ids[:num_tokens]


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def run_scenario(
    model,
    tokenizer,
    budget: int,
    victim_prompt: List[int],
    long_prompt: List[int],
    victim_tokens: int,
    warmup_steps: int,
    block_size: int,
    kv_budget_mb: float,
) -> Dict[str, Any]:
    """
    One run: a victim decoding steadily, then a long prompt lands on it.

    Returns the victim's inter-token gaps, which is the only measurement that
    matters here -- the intruder's own latency is expected to be worse under
    chunking and is reported for honesty, not as a target.
    """
    engine = ContinuousBatchingEngine(
        model,
        tokenizer,
        block_size=block_size,
        kv_budget_mb=kv_budget_mb,
        max_num_seqs=4,
        max_model_len=len(long_prompt) + victim_tokens + 64,
        enable_prefix_sharing=False,  # isolate the scheduling effect
        max_prefill_tokens_per_step=budget,
    )
    params = SamplingParams(
        max_new_tokens=victim_tokens, temperature=0.0, stop_on_eos=False
    )

    victim = engine.submit(victim_prompt, params)
    victim_token_times: List[float] = []
    intruder_submitted_at = None
    intruder = None
    intruder_first_token_at = None

    steps = 0
    while engine.has_work() and steps < 20000:
        report = engine.step()
        steps += 1
        now = time.perf_counter()
        for event in report.events:
            if event.seq_id == victim:
                victim_token_times.append(now)
            elif intruder is not None and event.seq_id == intruder:
                if intruder_first_token_at is None:
                    intruder_first_token_at = now

        # Drop the long prompt in once the victim is decoding steadily, so the
        # gaps before and after are comparable.
        if intruder is None and len(victim_token_times) >= warmup_steps:
            intruder_submitted_at = time.perf_counter()
            intruder = engine.submit(
                long_prompt,
                SamplingParams(max_new_tokens=2, temperature=0.0, stop_on_eos=False),
            )

    gaps = [
        (b - a) * 1000.0
        for a, b in zip(victim_token_times, victim_token_times[1:])
    ]
    # Split at the arrival so "before" is a clean baseline for this very run,
    # which removes machine-to-machine and run-to-run variation from the claim.
    before, after = [], []
    for (a, b) in zip(victim_token_times, victim_token_times[1:]):
        gap = (b - a) * 1000.0
        if intruder_submitted_at is not None and b <= intruder_submitted_at:
            before.append(gap)
        else:
            after.append(gap)

    median_before = statistics.median(before) if before else 0.0
    return {
        "budget": budget,
        "victim_tokens": len(victim_token_times),
        "steps": steps,
        "gaps_ms": [round(g, 3) for g in gaps],
        "median_before_ms": round(median_before, 3),
        "median_after_ms": round(statistics.median(after), 3) if after else 0.0,
        "p99_ms": round(percentile(gaps, 99), 3),
        "max_ms": round(max(gaps), 3) if gaps else 0.0,
        "stall_ratio": round(max(gaps) / median_before, 2) if median_before else 0.0,
        "p99_ratio": round(percentile(gaps, 99) / median_before, 2) if median_before else 0.0,
        "intruder_ttft_ms": (
            round((intruder_first_token_at - intruder_submitted_at) * 1000.0, 1)
            if intruder_first_token_at and intruder_submitted_at
            else None
        ),
        "prefill_chunks": engine.telemetry()["prefill_chunks"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="MemForge Phase C chunked prefill")
    parser.add_argument("--model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--prompt-tokens", type=int, default=2048,
                        help="length of the intruding prompt")
    parser.add_argument("--victim-prompt-tokens", type=int, default=32)
    parser.add_argument("--victim-tokens", type=int, default=24,
                        help="tokens the in-flight request generates")
    parser.add_argument("--warmup-tokens", type=int, default=4,
                        help="victim tokens before the long prompt arrives")
    parser.add_argument("--budgets", type=int, nargs="+", default=[0, 512, 256, 128],
                        help="prefill token budgets per step; 0 means unlimited")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--kv-budget-mb", type=float, default=1024.0)
    args = parser.parse_args()

    banner("MemForge Phase C — does a long prompt stall in-flight decoding?")
    print(f"model            : {args.model}")
    print(f"intruding prompt : {args.prompt_tokens} tokens")
    print(f"victim           : {args.victim_prompt_tokens}-token prompt, "
          f"{args.victim_tokens} tokens generated")
    print(f"budgets          : {args.budgets}  (0 = unlimited, the old behaviour)")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    long_prompt = build_prompt(tokenizer, args.prompt_tokens)
    victim_prompt = build_prompt(tokenizer, args.victim_prompt_tokens)

    rows: List[Dict[str, Any]] = []
    for budget in args.budgets:
        label = "unlimited" if budget == 0 else str(budget)
        banner(f"prefill budget: {label} tokens/step")
        row = run_scenario(
            model, tokenizer, budget, victim_prompt, long_prompt,
            args.victim_tokens, args.warmup_tokens, args.block_size, args.kv_budget_mb,
        )
        rows.append(row)
        print(f"  victim tokens          : {row['victim_tokens']}")
        print(f"  median gap before      : {row['median_before_ms']} ms")
        print(f"  median gap after       : {row['median_after_ms']} ms")
        print(f"  p99 gap                : {row['p99_ms']} ms  "
              f"({row['p99_ratio']}x the victim's own median)")
        print(f"  worst gap              : {row['max_ms']} ms  "
              f"({row['stall_ratio']}x)")
        print(f"  prefill chunks         : {row['prefill_chunks']}")
        print(f"  intruder TTFT          : {row['intruder_ttft_ms']} ms")

    banner("Summary")
    print(f"{'budget':>10}{'median before':>15}{'p99':>10}{'p99 x':>8}"
          f"{'worst':>10}{'worst x':>10}{'chunks':>8}")
    for row in rows:
        label = "unlimited" if row["budget"] == 0 else str(row["budget"])
        print(f"{label:>10}{row['median_before_ms']:>15.2f}{row['p99_ms']:>10.2f}"
              f"{row['p99_ratio']:>8.2f}{row['max_ms']:>10.2f}"
              f"{row['stall_ratio']:>10.2f}{row['prefill_chunks']:>8}")

    baseline = next((r for r in rows if r["budget"] == 0), None)
    best = min((r for r in rows if r["budget"] > 0),
               key=lambda r: r["stall_ratio"], default=None)
    verdict = None
    if baseline and best:
        improvement = baseline["max_ms"] / best["max_ms"] if best["max_ms"] else 0.0
        print(f"\nWorst-case stall: {baseline['max_ms']:.1f} ms unlimited vs "
              f"{best['max_ms']:.1f} ms at budget {best['budget']} "
              f"({improvement:.1f}x better)")

        # "Unaffected" means the worst gap sits near the victim's own median
        # rather than at a multiple of it. Chunking alone cannot reach that,
        # and the reason is worth stating rather than reporting a bare failure:
        # a step that carries a prefill chunk runs TWO forward passes -- one for
        # the chunk, one for the decode batch -- so the gap floors at roughly 2x
        # the median however small the chunk gets. Shrinking the budget past
        # that point buys nothing and costs prefill throughput, which the
        # intruder TTFT column shows directly.
        met = best["stall_ratio"] <= 2.0
        floor_reached = (
            len([r for r in rows if r["budget"] > 0]) >= 2
            and best["stall_ratio"] <= 3.0
        )
        verdict = (
            f"Phase C bullet 1 (chunked prefill): worst-case stall cut "
            f"{improvement:.1f}x, from {baseline['stall_ratio']}x the victim's "
            f"median to {best['stall_ratio']}x at budget {best['budget']}."
        )
        if met:
            verdict += " Target met: in-flight decoding is effectively unaffected."
        elif floor_reached:
            verdict += (
                " Target NOT met, and the residual is understood: the ratio "
                "asymptotes near 2x because a step carrying a chunk runs two "
                "forward passes (chunk, then decode batch). Removing that "
                "requires Phase C bullet 2, prefill/decode fusion into a single "
                "pass, which is not implemented."
            )
        else:
            verdict += " Target NOT met; try smaller budgets to find the floor."
        print(verdict)

    payload = {
        "benchmark": "chunked_prefill",
        "phase": "C",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": {
            "system": platform.system(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "config": {
            "model": args.model,
            "prompt_tokens": args.prompt_tokens,
            "victim_prompt_tokens": args.victim_prompt_tokens,
            "victim_tokens": args.victim_tokens,
            "budgets": args.budgets,
            "block_size": args.block_size,
        },
        "verdict": verdict,
        "rows": rows,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
