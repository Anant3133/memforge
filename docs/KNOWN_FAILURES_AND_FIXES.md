# What is failing, why, and what we did about it

Measured on the A100 server, 2026-11-01. Every number traces to a JSON/log under
`benchmarks/results/runs/<label>/`; `benchmarks/results/runs/AGGREGATE.md` has the
median and min-max over independent rounds. Where this contradicts an older
document, this one is newer and has the data.

Status key: ✅ fixed and tested · 🟡 improved, not solved · ❌ open

---

## 1. Speculative decoding was a net loss everywhere — 🟡 batch penalty removed, no clear win on a contended card

**Symptom.** Speculation was slower than plain decoding on every pair and batch
size on the GPU: 0.3×–0.7×. Qwen3-8B ← 0.6B at batch 1 ran at **0.70×**.

**What it was not.** Not the draft's *guess quality*. Acceptance was 0.45–0.70, and
even Qwen3-4B as the draft (acceptance 0.70) only reached 0.73×.

**Root cause (measured).** An eager Hugging Face decode step at batch 1 is bound by
Python and kernel-launch overhead, not arithmetic:

| model | ms / token (wall) | ms / token (GPU actually busy) |
|---|---:|---:|
| Qwen3-0.6B (draft) | 28.8 | 7.6 |
| Qwen3-1.7B | 34.6 | 9.6 |
| Qwen3-8B (target) | 45.3 | 18.2 |

The "small" draft cost **63 % of the target's step**, so K serial draft steps
swallowed everything verification saved. Cost model, fed with the measured numbers,
predicted 0.69× before any fix; the benchmark measured 0.70×.

**Fix.** The draft's single-token step is captured as a CUDA graph and replayed
(`GraphedDraftPool` in `memforge_ml/runtime/batched_speculative.py`, enabled with
`ContinuousBatchingEngine(draft_cuda_graph=True)` / `--draft-cuda-graph`). One
replay costs **6.5 ms/token** instead of 28.8.

**Result** (3 rounds, eager and graphed run back to back, fp16, 32 new tokens):

| pair, K | batch | eager | graphed |
|---|---:|---:|---:|
| Qwen3-8B ← 0.6B, K=3 | 1 | 0.698 | **1.029** [1.008–1.034] |
| | 2 | 0.576 | 0.886 |
| | 4 | 0.332 | 0.589 |
| Qwen3-8B ← 0.6B, K=4 | 1 | 0.656 | **1.037** [1.030–1.037] |
| Qwen3-8B ← 1.7B, K=3 | 1 | 0.694 | 0.965 |
| Qwen2.5-7B ← 0.5B, K=3 | 1 | 0.665 | 0.894 |

* Acceptance is **identical** between the eager and graphed arms (0.5455 both):
  the graphed draft proposes exactly the same tokens; only the cost changed.
* The graph is faster in **every** configuration (~1.3–1.7× over eager).
* It produces an outright win **only at batch 1 with the smallest draft**, and by
  a small margin (~3 %). Be honest about that: this is "break-even to slightly
  ahead", not the 2–3× the plan promised.

**Batch ≥ 2: fixed in shape, not yet a win 🟡.** The per-sequence graph still ran
the draft once per sequence, so draft cost grew with the batch. `BatchedGraphedDraftPool`
(`--draft-cuda-graph batched`) keeps one KV cache with a row per sequence and drafts
the whole batch in K+1 replays regardless of its size. Rows sit at different
positions in one replay (a per-row write index and an in-graph causal mask).
Qwen3-8B ← 0.6B, K=3, median of 3 rounds, arms run back to back:

| batch | eager draft | per-sequence graph | **batched graph** |
|---:|---:|---:|---:|
| 1 | 0.700 | 0.782 | **0.821** |
| 2 | 0.637 | 0.729 | **0.911** |
| 4 | 0.413 | 0.505 | **0.816** |
| 8 | 0.353 | 0.434 | **0.861** |

The batch penalty is gone (the curve is flat instead of collapsing), and at batch 8
it is 2.4x the eager draft. Qwen2.5-7B ← 0.5B behaves the same (0.83 / 0.82 / 0.78 / 0.89).
Output equivalent in every run; acceptance is identical to the eager draft.
**It still does not beat plain decoding (all < 1.0) on this card.** The earlier
1.03x at batch 1 was measured when the card was quiet; these rounds ran next to
other tenants at 100 % utilisation, which slows every kernel of the draft.

*Negative result:* unrolling the whole greedy round (K+1 steps and the argmax feedback)
into a single replay was *not* faster (0.64-0.85, below the stepwise batched graph).
A direct profile shows why: a round costs ~70 ms either way at batch 1 and ~117 ms at
batch 8, so on this shared card the draft is bound by GPU time, not launches. Kept as
an option (`batched-unrolled`), not recommended.

**Dead ends measured, so nobody repeats them:**
* Bigger draft (1.7B, 4B): higher acceptance, but a slower draft; net worse.
* Longer lookahead: K=2/3 best (0.67–0.76× eager), K=6/8 collapse to 0.39–0.53×
  because acceptance falls faster than the extra guesses pay.

---

## 2. Qwen2.5-7B could not use a smaller Qwen2.5 draft — ✅ fixed

**Symptom.** `RuntimeError: The size of tensor a (152064) must match the size of
tensor b (151936)` in the batched speculative path.

**Cause.** Same tokenizer, different *padded* vocabulary: the 7B has 152064 logit
rows, the 0.5B/1.5B/3B have 151936. The runbook's "all Qwen2.5 share a vocab" is
true only up to 3B. The non-batched runtime had a guard; the batched path did not.

**Fix.** `resolve_round` restricts the target's distribution to the width the draft
covers before the softmax. The extra rows are padding no tokenizer can emit.

**Test.** `tests/python/test_vocab_padding.py`. The first version of this test
*passed on the broken code* (it never reached the rejection path); the second one
reproduces the exact crash without the fix and passes with it.

---

## 3. fp16 speculative output not byte-identical on some models — ✅ explained, check automated

**Symptom.** "Lossless: False" appeared intermittently in fp16 (Qwen3-8B).

**Test, not assertion.** `memforge_ml/eval/divergence.py` takes the *first* position
where speculative and plain output disagree. The context there is identical, so one
clean single-sequence forward gives the logits both runs were argmax-ing; it reports
the gap between the two chosen tokens against a tolerance scaled to the dtype's
rounding error. A mismatch with a gap inside the tolerance is a near-tie; a token far
below the top would be flagged as a real bug. The speculative benchmark now runs this
automatically and records `output_equivalent`.

**Result (60 benchmark rows, Qwen3-8B and Qwen2.5-7B pairs, fp16):** 42 byte-identical;
18 differed; **all 18 explained**. In the cases inspected the single-sequence logit gap
between the two tokens was exactly **0.0**: fp16 gave two tokens the same score and the
tie broke differently depending on batch shape. No unexplained divergence. Byte-identity
therefore holds in fp32 and is replaced by "identical or an explained exact tie" in
half precision. Unit tests (`test_divergence.py`) cover near-tie / real-bug / length cases.

---

## 4. Paged attention vs contiguous on this GPU — 🟡 now beats stock decoding at most sizes; 2048 on big models still short

**Root cause.** Launch overhead, not bandwidth: per layer the eager paged path adds an
index translation, a K write, a V write, a K gather and a V gather, each a Python-launched
kernel on a step that is already launch-bound.

**Fix.** `GraphedPagedDecoder` (`memforge_ml/runtime/graphed_paged_decode.py`, engine
option `paged_cuda_graph=True`): index translation, the new token's KV write *into its
physical block*, the context gather, the causal mask and the forward are one captured
graph. Tokens, positions and block tables are device-tensor inputs; nothing host-derived
is baked in. Graphs are bucketed by batch (power of two, padded by repeating row 0, which
rewrites identical values) and context (32-token buckets).

Paged-graphed throughput as a multiple of **stock eager contiguous** decoding, median of
3 rounds (before = eager paged; target >= 0.95). Output identical to contiguous in every cell:

| model | ctx 128 | 512 | 1024 | 2048 | before @2048 |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 1.67 | 1.40 | 1.30 | 1.05 | 0.79 |
| Qwen2.5-1.5B | 1.29 | 1.24 | 1.13 | 1.01 | 0.80 |
| Qwen2.5-3B | 1.26 | 1.18 | 1.10 | **0.96** | 0.83 |
| Qwen2.5-7B | 1.05 | 1.05 | 1.02 | 0.93 | 0.92 |
| Qwen3-8B | 1.12 | 1.03 | 0.97 | 0.92 | 0.92 |

(128/512 and 0.5B/1.5B columns use the 128-token gather bucket; 3B/7B/8B at 1024/2048
use the 32-token bucket, which lifted 3B@2048 from 0.92 to 0.96.)

**Verdict.** The target is met everywhere except 2048 tokens on 7B and 8B (0.93, 0.92).
Against an equally graphed contiguous baseline paged is 0.65-0.96x, so the remaining gap
is real memory traffic from the per-layer gather, growing with context. **Open ❌:** read
KV from the pool inside attention (block-wise kernel) instead of gathering a copy.

---

## 5. Chunked prefill: works; CPU target met on the median for 4 of 5 models, GPU just short — 🟡

Definition of done: the worst decode stall caused by an incoming long prompt
≤ 2.0× a normal step. Prompt 512 tokens, median of 3 rounds:

| | unchunked | chunked, budget 4 | chunked, budget 16 |
|---|---:|---:|---:|
| GPU, Qwen2.5-7B | 3.0 | **2.2** | 2.2 |
| GPU, Qwen3-8B | 2.6 | **2.3** | 2.2 |
| GPU, SmolLM2-360M | 3.1 | 2.2 | 2.4 |
| CPU, Qwen2.5-7B (1 run) | 6.0 | **1.78** | 4.4 |
| CPU, Qwen2.5-1.5B (1 run) | 8.2 | **1.73** | 4.0 |

* **CPU, 4 runs per model, budget 4:** median stall 1.81 (0.5B), 1.96 (1.5B), 1.84 (3B),
  1.82 (7B), 2.10 (SmolLM2-360M). Under 2.0 on the median for 4 of 5, but **not in every
  run** (ranges reach 2.05-2.59). The earlier single-run "target met" was optimistic.
* **GPU: a real 25–30 % reduction, floor ≈ 2.15×, target not met.** The floor is
  independent of model size, which points at a fixed per-step cost, not at the
  prefill chunk.
* **Fusion makes the GPU worse, not better.** Fused vs unfused, 3 rounds:
  budget 4: 2.4 vs 2.15 (1.5B), 2.4 vs 2.2 (7B); budget 64: 5.5–5.7 vs 2.1–2.3.
  This reverses the runbook's "fusion helps at every budget" (that was CPU).
  Recommendation: leave `fused_prefill` off on GPU.

---

## 5b. Qwen3-14B (downloaded earlier, now run) — ✅

fp16 needs ~29.5 GB; the shared card had ~25 GB free. `memforge_ml/models/int8_linear.py`
quantises every Linear to int8 (per-output-channel, scale applied to the output) straight
onto the GPU, so the fp16 model never has to fit. It changes numerics, so only comparisons
*within* the int8 model are valid, which is all the checks need.

* GPU int8, 3 rounds: paged / contiguous = **0.988 / 0.987 / 0.991 / 0.988** at 128 / 512 /
  1024 / 2048 tokens (meets 0.95 even without graphs: the dequantising step is expensive
  enough to hide the paging overhead), output identical; prefix sharing 0.333 blocks, 0.27
  prefill, 0 leaks; chunked-prefill stall 2.63 -> 2.03 at budget 4 (target 2.0, just over).
* CPU fp32, 1 run: paged 1.00x, prefix 0.333, stall 6.77 -> 1.98.

Open: fp16 14B (needs ~30 GB free); speculation on 14B not run.

---

## 6. Things that now work and were not shown before

| claim | evidence |
|---|---|
| Paged output = contiguous output, on 7 models up to 8B | every context-sweep row, 3 rounds |
| Prefix sharing: 4 requests → 0.333× blocks, 0.27× prefill work, 0 leaks, identical output | 7 models × 3 rounds, identical every time |
| Soak: 110 requests, concurrency 8, no leaks, RSS flat | 0.5B, 1.5B, 3B, 7B, Qwen3-8B: 0 leaked blocks; RSS growth 3.7–25 MB vs 50 MB limit |
| Test suite | CPU: 417 passed, 20 skipped (GPU-only), 3 consecutive runs; GPU: 27 passed, 3 consecutive runs |
| Regression gate | `scripts/check_targets.py`: 0 hard failures; fails on output mismatch, leaked blocks, unexplained speculative divergence, or a >5% metric drop |

---

## 7. Process lessons (each one produced a wrong result at some point)

1. **A test that cannot fail is worthless.** Two of my new tests passed on broken
   code. Inject the bug, watch it fail, then trust it. The graphed-draft test only
   became meaningful once it compared *which tokens the draft proposed*, because
   speculation stays lossless however bad the draft is.
2. **Library internals move.** transformers 5.x `StaticLayer` ignores
   `cache_position` and keeps its own cursor. Graph capture, warmup and replay all
   moved it, so the first graphed draft was silently wrong (logits off by 16).
3. **Capture at a harmless position.** Warmup executes real steps; running it at
   position 0 corrupted the prompt's first KV entry.
4. **Measure the cost model before optimising.** Four draft sizes and four
   lookaheads changed nothing; one timing probe (§1) found the cause.
5. **Several benchmark scripts print the wrong output path** ("Wrote
   benchmarks/results/x.json") even when `--out` sent the file elsewhere; and
   `run_stress_test.py` ignores `--out` and overwrites a tracked file.
