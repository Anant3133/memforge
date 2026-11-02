# MemForge tasks and backlog

Status: ✅ done and tested · 🟡 partly done · ⏳ in progress · ❌ not started · 🚫 blocked (reason given)

Evidence for every ✅ is in `benchmarks/results/runs/` and summarised in
[RESULTS_REPORT.md](RESULTS_REPORT.md) and [KNOWN_FAILURES_AND_FIXES.md](KNOWN_FAILURES_AND_FIXES.md).

## A. The six improvement items

| # | Task | Status | Evidence / note |
|---|---|---|---|
| A1 | Batched CUDA-graph draft | ✅ | batch collapse removed (0.35 -> 0.86 at batch 8); still < 1.0x on a contended card |
| A2 | CUDA-graphed paged decode | ✅ | 0.92-1.67x stock; misses 0.95 at 2048 tokens on 7B/8B |
| A3 | fp16/bf16 equivalence check | ✅ | 18/18 mismatches in 60 runs are exact logit ties |
| A4 | Repeat CPU chunked-prefill | ✅ | 4 runs/model; median < 2.0 for 4 of 5 models, not in every run |
| A5 | Quiet-GPU rerun | 🚫 | card is shared and at 100% utilisation; GPU numbers labelled contended |
| A6 | Qwen3-14B | ✅ | GPU (int8 weights, 3 runs): paged/contiguous 0.99 at 128-2048 tokens, identical output, prefix sharing 0.333, chunked stall 2.03x at budget 4; CPU fp32 (1 run): 1.00x, 1.98x |

## B. Active tasks (this session)

| # | Task | Status |
|---|---|---|
| T1 | int8 weight-only loader so 14B fits the 25 GB left on the card; 14B GPU runs | ✅ `memforge_ml/models/int8_linear.py`, 5 tests; 3 GPU rounds |
| T2 | Finish 14B CPU chunked-prefill run and record it | ✅ |
| T3 | Regression gate (`scripts/check_targets.py`, baseline `benchmarks/targets_baseline.json`) | ✅ 6 tests; 0 hard failures on current runs |
| T4 | One-command setup script (`scripts/setup_server.sh`, has `--dry-run`) | ✅ syntax + dry run checked; not executed (network was down) |
| T5 | Re-run all GPU tests after the 32-token bucket default; repeat 3x | ✅ 27 passed x3 |
| T6 | Re-run the full CPU suite; repeat 3x | ✅ 417 passed, 20 skipped x3 |
| T7 | Final results report + docs refresh, commit | ✅ |

## C. Backlog (not started, ordered by expected payoff)

| # | Task | Why / expected gain |
|---|---|---|
| B1 | Read KV from the pool inside attention instead of gathering a copy | closes the 2048-token paged gap on 7B/8B (0.92-0.93 -> >= 0.95) |
| B2 | Make speculation win: larger target, tuned draft, rerun on a quiet card | batch-1 1.03x was seen on a quiet card; nothing wins on a contended one |
| B3 | Make the graphed paths default on GPU with a safe eager fallback | users get the speedup without flags |
| B4 | Chunked prefill on GPU: floor is ~2.15x (target 2.0x) | per-step fixed cost; try graphing the chunk step |
| B5 | Fused prefill makes GPU slower: fix or remove | currently off by default |
| B6 | Speculation with sampling: unrolled draft is greedy-only | sampled output only has the statistical check |
| B7 | Retire one of the two graphed-draft pools | per-sequence pool is superseded by the batched one |
| B8 | Push to GitHub | needs a fresh access token from the owner |
| B9 | Confirm whose GPU miner was seen earlier in this container | owner question; it is no longer running |
| B10 | Run 14B in fp16 (not int8) | needs ~30 GB free; the other tenants hold 15.6 GB |
| B11 | Re-run pre-explainer speculative runs so the gate can classify them | older rows have no `output_equivalent` |

## D. Known limits (not bugs)

* GPU graphs need `attn_implementation="sdpa"` and a CUDA device.
* int8 weights (T1) change numerics: comparisons are paged-vs-contiguous *within* the int8 model, never against fp16.
* Absolute GPU tokens/s are not comparable across runs on this shared card; ratios between arms run back to back are.
