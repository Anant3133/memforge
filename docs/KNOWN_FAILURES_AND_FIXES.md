# What is failing, why, and what we did about it

Measured on the A100 server, 2026-11-01. Every number traces to a JSON/log under
`benchmarks/results/runs/<label>/`; `benchmarks/results/runs/AGGREGATE.md` has the
median and min-max over independent rounds. Where this contradicts an older
document, this one is newer and has the data.

Status key: ✅ fixed and tested · 🟡 improved, not solved · ❌ open

---

## 1. Speculative decoding was a net loss everywhere — 🟡 now wins in one case

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

**Still open ❌ — batch ≥ 2 still loses (0.5–0.9×).** The draft runs once per
sequence per step, so draft cost grows with the batch while verification does not.
*Next step:* one graph with a batch dimension, so B sequences draft in a single
replay. Expected to remove the batch penalty; not yet built.

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

## 3. fp16 speculative output not byte-identical on some models — ❌ open, scoped

**Symptom.** "Lossless at every batch size: False" appears intermittently in fp16 on
Qwen3-8B (and a few batch sizes of Qwen3-8B/1.7B pairs), in **both** the eager and
graphed arms. Never seen on Qwen2.5-7B or any `head_dim` 64 model.

**What we know.**
* It is not the graphed draft: it occurs identically with the eager draft.
* It is not an algorithm bug: speculation output equals plain output in fp32.
  On Qwen3-4B, fp16 and fp32 were both exact, **bf16 was not**. So it depends on
  dtype × model, not on "fp16 is bad".
* The pattern is consistent with the already-documented fact that batched and
  unbatched decode are different reduction orders in half precision, so greedy
  argmax can flip on near-ties.

**What to do.** Treat byte-identity as an fp32 guarantee. For fp16/bf16, assert
*statistical* equivalence (same token distribution) or compare against a plain run
that uses the same batch shape. Not yet implemented.

---

## 4. Paged attention is slower than contiguous on this GPU — 🟡 improves with size, never reaches the target

Definition of done: within 5 % of contiguous at 2048 tokens. Median over 3 rounds:

| model | 128 | 512 | 1024 | 2048 |
|---|---:|---:|---:|---:|
| SmolLM2-135M / 360M | 0.85 | 0.85 | 0.85 | 0.85 |
| Qwen2.5-0.5B | 0.85 | 0.85 | 0.85 | 0.86 |
| Qwen2.5-1.5B | 0.86 | 0.86 | 0.86 | 0.88 |
| Qwen2.5-3B | 0.86 | 0.86 | 0.85 | 0.88 |
| Qwen2.5-7B | 0.86 | 0.88 | 0.90 | **0.94** |
| Qwen3-8B | 0.89 | 0.90 | 0.90 | **0.92** |

**Verdict: not met on the A100 at any size we ran** (best is 0.94 on the 7B at
2048). The older "met at 2048" was measured on a laptop GPU (different hardware, `--kv-budget-mb 256`; this campaign used 1024). We did not isolate which difference matters.
Output was byte-identical to contiguous in every one of these runs.

The shape matters: small models sit on a flat 0.85 (a fixed per-step cost the
paging adds), and the gap closes only as the forward pass gets expensive enough to
amortise it. On **CPU** the same engine is at parity (0.96–1.03, single runs).

**Fix path ❌.** Fuse the block-table gather into the attention kernel so a decode
step does not pay a separate gather launch (the runbook's "real optimisation
target nobody has touched"). The graph-capture result in §1 suggests launch
overhead is the likely cost; capturing the *paged* decode step in a CUDA graph is
the cheapest experiment to try first.

---

## 5. Chunked prefill: works, but stops just short of the target on GPU — 🟡

Definition of done: the worst decode stall caused by an incoming long prompt
≤ 2.0× a normal step. Prompt 512 tokens, median of 3 rounds:

| | unchunked | chunked, budget 4 | chunked, budget 16 |
|---|---:|---:|---:|
| GPU, Qwen2.5-7B | 3.0 | **2.2** | 2.2 |
| GPU, Qwen3-8B | 2.6 | **2.3** | 2.2 |
| GPU, SmolLM2-360M | 3.1 | 2.2 | 2.4 |
| CPU, Qwen2.5-7B (1 run) | 6.0 | **1.78** | 4.4 |
| CPU, Qwen2.5-1.5B (1 run) | 8.2 | **1.73** | 4.0 |

* **CPU: target met** at budget 4 on every Qwen size (one run each — confirm with
  repeats before quoting).
* **GPU: a real 25–30 % reduction, floor ≈ 2.15×, target not met.** The floor is
  independent of model size, which points at a fixed per-step cost, not at the
  prefill chunk.
* **Fusion makes the GPU worse, not better.** Fused vs unfused, 3 rounds:
  budget 4: 2.4 vs 2.15 (1.5B), 2.4 vs 2.2 (7B); budget 64: 5.5–5.7 vs 2.1–2.3.
  This reverses the runbook's "fusion helps at every budget" (that was CPU).
  Recommendation: leave `fused_prefill` off on GPU.

---

## 6. Things that now work and were not shown before

| claim | evidence |
|---|---|
| Paged output = contiguous output, on 7 models up to 8B | every context-sweep row, 3 rounds |
| Prefix sharing: 4 requests → 0.333× blocks, 0.27× prefill work, 0 leaks, identical output | 7 models × 3 rounds, identical every time |
| Soak: 110 requests, concurrency 8, no leaks, RSS flat | 0.5B, 1.5B, 3B, 7B, Qwen3-8B: 0 leaked blocks; RSS growth 3.7–25 MB vs 50 MB limit |
| Test suite | 400 passed, 2 skipped (GPU-only) on CPU; the GPU tests pass on the A100 |

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
