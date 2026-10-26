#!/usr/bin/env python3
"""
Phase D benchmark: do speculation and continuous batching actually compose?

Phase D's definition of done has two halves, and they need different machines:

  * *"speculation and batching compose"* -- a structural claim about how many
    target forward passes a round costs. Measurable here, on CPU, because it is
    a count, not a time.
  * *"a GPU run either confirms or refutes the 2-3x claim"* -- a wall-clock
    claim. **Not measurable here**: `torch.cuda.is_available()` is False, and
    the only model pair that fits on this CPU (a 360M target with a 135M draft)
    is too close in cost for speculation to pay. §3.5 of
    PROGRESS_AND_NEXT_PHASES.md already measured that at 0.58x -- slower.

So this script measures the first half honestly and refuses to fake the second.

What it reports, per batch size:

  * **target passes per round.** Batched verification spends one target pass per
    round regardless of batch size. A per-sequence verifier would spend one per
    (sequence, round) pair. The ratio between them is the saving, and it is the
    whole reason the two features can be used together: without it, speculating
    over a batch of N costs N times the target work and is strictly worse than
    plain batched decoding.
  * **tokens per target pass.** The throughput-relevant figure: how much output
    one pass of the expensive model buys.
  * **wall-clock tok/s, both ways.** Reported even though it is expected to be
    unfavourable on this hardware, because leaving it out would be the
    dishonest part.

Usage:
    python benchmarks/run_speculative_batching.py
    python benchmarks/run_speculative_batching.py --batch-sizes 1 2 4 8 --lookahead 4
    python benchmarks/run_speculative_batching.py --self-draft   # forces full acceptance
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from memforge_ml.model_defaults import (  # noqa: E402
    DEFAULT_DRAFT_MODEL,
    DEFAULT_TARGET_MODEL,
)
from memforge_ml.runtime.continuous_engine import ContinuousBatchingEngine  # noqa: E402
from memforge_ml.runtime.sampling import SamplingParams  # noqa: E402

RESULTS_PATH = PROJECT_ROOT / "benchmarks" / "results" / "speculative_batching.json"

PROMPT_POOL = [
    "Explain how paged attention avoids fragmenting a key-value cache.",
    "Describe continuous batching and why it beats static batching.",
    "What problem does speculative decoding solve, and what does it cost?",
    "Summarise copy-on-write as it applies to shared prefix blocks.",
    "Why does a block-level prefix hash have to be chained?",
    "How does chunked prefill protect an in-flight request's latency?",
    "What makes rejection sampling lossless rather than approximate?",
    "Explain the difference between a logical and a physical block table.",
]


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def run_scenario(
    model,
    tokenizer,
    draft_model,
    lookahead: int,
    prompts: List[str],
    max_new_tokens: int,
    block_size: int,
    kv_budget_mb: float,
) -> Dict[str, Any]:
    """One generation run, with speculation on (draft_model given) or off."""
    engine = ContinuousBatchingEngine(
        model,
        tokenizer,
        block_size=block_size,
        kv_budget_mb=kv_budget_mb,
        max_num_seqs=max(len(prompts), 1),
        max_model_len=1024,
        draft_model=draft_model,
        lookahead_k=lookahead if draft_model is not None else 0,
    )
    # `stop_on_eos=False` so every sequence decodes the same fixed length.
    # With stopping on, this model ends most of these prompts after one or two
    # tokens, which leaves no concurrent decode to batch and makes every
    # per-round figure below meaningless -- the first version of this benchmark
    # reported 1.0 sequences per round for exactly that reason.
    params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        stop_on_eos=False,
    )

    start = time.perf_counter()
    results = engine.generate(prompts, params)
    elapsed = time.perf_counter() - start

    telemetry = engine.telemetry()
    tokens = sum(r["generated_tokens"] for r in results)
    rounds = telemetry["speculative_rounds"]
    sequence_rounds = engine.total_speculative_sequence_rounds

    return {
        "tokens": tokens,
        "elapsed_s": round(elapsed, 3),
        "tokens_per_s": round(tokens / elapsed, 2) if elapsed else 0.0,
        "target_forward_passes": engine.total_forward_passes,
        "speculative_rounds": rounds,
        # What a per-sequence verifier would have cost for the same rounds.
        "naive_target_passes": sequence_rounds,
        "sequences_per_round": telemetry["speculative_sequences_per_round"],
        "target_passes_saved": max(0, sequence_rounds - rounds),
        "tokens_per_target_pass": (
            round(tokens / engine.total_forward_passes, 3)
            if engine.total_forward_passes
            else 0.0
        ),
        "draft_acceptance_rate": telemetry["draft_acceptance_rate"],
        "token_ids": [r["token_ids"] for r in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument(
        "--self-draft",
        action="store_true",
        help="use the target as its own draft; forces 100%% greedy acceptance, "
             "which isolates the structural saving from the draft's quality",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--kv-budget-mb", type=float, default=512.0)
    args = parser.parse_args()

    if max(args.batch_sizes) > len(PROMPT_POOL):
        parser.error(
            f"batch size {max(args.batch_sizes)} exceeds the "
            f"{len(PROMPT_POOL)} distinct prompts available"
        )

    banner("Phase D - speculative decoding inside continuous batching")
    print(f"target model : {args.model}")
    print(f"draft model  : {'(self)' if args.self_draft else args.draft_model}")
    print(f"lookahead K  : {args.lookahead}  (scheduler reserves K+1 = "
          f"{args.lookahead + 1} slots per sequence per step)")
    print(f"device       : cpu  (cuda available: {torch.cuda.is_available()})")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    if args.self_draft:
        draft_model = model
    else:
        draft_model = AutoModelForCausalLM.from_pretrained(
            args.draft_model, dtype=torch.float32
        )
        draft_model.eval()

    rows: List[Dict[str, Any]] = []
    for batch in args.batch_sizes:
        prompts = PROMPT_POOL[:batch]
        banner(f"batch = {batch}")

        plain = run_scenario(
            model, tokenizer, None, 0, prompts,
            args.max_new_tokens, args.block_size, args.kv_budget_mb,
        )
        spec = run_scenario(
            model, tokenizer, draft_model, args.lookahead, prompts,
            args.max_new_tokens, args.block_size, args.kv_budget_mb,
        )

        # Losslessness is the precondition for every other number here being
        # meaningful: a faster engine that changes the output is not faster.
        identical = plain["token_ids"] == spec["token_ids"]
        print(f"  output identical to non-speculative : {identical}")
        if not identical:
            print("  !! speculation changed the output -- the numbers below are void")

        print(f"  tokens generated                    : {spec['tokens']}")
        print(f"  speculative rounds                  : {spec['speculative_rounds']}")
        # Verification passes only. `target_forward_passes` also counts prefill,
        # so comparing it against the naive verification count would be
        # comparing different things -- which the first version of this
        # benchmark did, reporting a saving in 1 of 3 configurations instead
        # of 2 of 2.
        print(f"  verify passes (batched)             : {spec['speculative_rounds']}")
        print(f"  verify passes (per-sequence)        : {spec['naive_target_passes']}")
        print(f"  verify passes saved                 : {spec['target_passes_saved']}")
        print(f"  all target passes (incl. prefill)   : {spec['target_forward_passes']}")
        print(f"  sequences verified per round        : {spec['sequences_per_round']}")
        print(f"  draft acceptance rate               : {spec['draft_acceptance_rate']}")
        print(f"  tokens per target pass  spec / plain: "
              f"{spec['tokens_per_target_pass']} / {plain['tokens_per_target_pass']}")
        print(f"  wall-clock tok/s        spec / plain: "
              f"{spec['tokens_per_s']} / {plain['tokens_per_s']}  "
              f"({spec['tokens_per_s'] / plain['tokens_per_s']:.2f}x)"
              if plain["tokens_per_s"] else "")

        rows.append({
            "batch_size": batch,
            "output_identical": identical,
            "speculative": {k: v for k, v in spec.items() if k != "token_ids"},
            "non_speculative": {
                k: v for k, v in plain.items() if k != "token_ids"
            },
            "speedup": (
                round(spec["tokens_per_s"] / plain["tokens_per_s"], 3)
                if plain["tokens_per_s"] else None
            ),
        })

    banner("Verdict")
    all_lossless = all(r["output_identical"] for r in rows)
    # A batch of one cannot save anything, so it is excluded from the
    # denominator rather than counted as a failure to compose.
    multi = [r for r in rows if r["batch_size"] > 1]
    composed = [
        r for r in multi
        if r["speculative"]["naive_target_passes"]
        > r["speculative"]["speculative_rounds"]
    ]
    verdict_lines = [
        f"Lossless at every batch size: {all_lossless}.",
        f"Batched verification saved verify passes in "
        f"{len(composed)}/{len(multi)} multi-sequence configurations "
        f"(a batch of one has nothing to batch).",
    ]
    best = max(
        (r["speculative"]["sequences_per_round"] for r in rows), default=0.0
    )
    verdict_lines.append(
        f"Peak sequences verified per target pass: {best}. A per-sequence "
        f"verifier would have needed that many passes per round, so this is "
        f"the factor by which speculation and batching compose rather than "
        f"conflict."
    )
    speedups = [r["speedup"] for r in rows if r["speedup"]]
    if speedups:
        verdict_lines.append(
            f"Wall-clock speedup on this CPU: "
            f"{min(speedups):.2f}x-{max(speedups):.2f}x. Values below 1.0 are "
            f"expected and are not a defect of the composition: a 360M target "
            f"and a 135M draft are too close in cost for speculation to pay, "
            f"and the draft's K passes are serial. The 2-3x claim in the plan "
            f"is a GPU claim about a pair with a real cost gap, and this "
            f"machine has no CUDA device, so it is neither confirmed nor "
            f"refuted here."
        )
    for line in verdict_lines:
        print(f"  - {line}")

    payload = {
        "benchmark": "speculative_batching",
        "phase": "D",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": {
            "system": platform.system(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "cuda_available": torch.cuda.is_available(),
        },
        "config": {
            "model": args.model,
            "draft_model": "(self)" if args.self_draft else args.draft_model,
            "lookahead_k": args.lookahead,
            "lookahead_slots_reserved": args.lookahead + 1,
            "batch_sizes": args.batch_sizes,
            "max_new_tokens": args.max_new_tokens,
            "block_size": args.block_size,
        },
        "verdict": verdict_lines,
        "rows": rows,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    return 0 if all_lossless else 1


if __name__ == "__main__":
    raise SystemExit(main())
