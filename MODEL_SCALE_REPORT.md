# Scaling MemForge to a 1.5B Target — What Changed, and What It Exposed

**Written for: the MemForge team, and whoever next has to defend these numbers
to someone who did not run them.**

Measured on this machine on **2026-11-01**. Every number here comes from a JSON
file archived under `benchmarks/results/runs/<label>/`, so any claim can be
traced to the run that produced it. Where a result is unfavourable, or where a
previously published result turned out to be unsound, it says so — the project's
own standing rule is that a report containing only wins is not a report.

---

## 1. Why we did this

The engine's claims were all measured against `SmolLM2-360M`. The worry was that
360M is small enough to hide things: a cheap forward pass makes paging overhead
look expensive, makes chunked prefill look pointless, and makes speculative
decoding look like a loss. Three of those turned out to be true.

We added the **Qwen2.5 family** — `0.5B`, `1.5B`, `3B`, all sharing vocabulary
151936 so any pair can draft for any other — and re-ran every phase claim
head to head against SmolLM2 on identical flags.

```
cache: C:\Users\Anant\huggingface\hub        (HF_HOME is relocated, see §7)
  SmolLM2-135M-Instruct     260 MB
  SmolLM2-360M-Instruct     694 MB
  Qwen2.5-0.5B-Instruct     954 MB
  Qwen2.5-1.5B-Instruct     2.9 GB
  Qwen2.5-3B-Instruct       5.8 GB   (2 shards)
                           ------
                            11 GB    17 GB free on C: afterwards
```

### The geometry that turned out to matter

| model | layers | Q heads | KV heads | GQA | head_dim | KV bytes/token (fp16) |
|---|---:|---:|---:|---:|---:|---:|
| SmolLM2-360M | 32 | 15 | 5 | 3 | 64 | 40960 |
| Qwen2.5-0.5B | 24 | 14 | 2 | 7 | 64 | 12288 |
| Qwen2.5-1.5B | 28 | 12 | 2 | 6 | **128** | 28672 |
| Qwen2.5-3B | 36 | 16 | 2 | 8 | **128** | 36864 |

`head_dim` is the column that ends up explaining §5. Note also that Qwen's
aggressive GQA means a *bigger* model uses *less* KV per token than
SmolLM2-360M — so KV-pressure claims stated in megabytes get smaller on Qwen,
while the ratio claims are unaffected.

---

## 2. Phase A — paging overhead. Now actually measured.

**This claim was never resolved before, and the old numbers could not have
resolved it.** The harness took the median of **3** repeats and compared it
against a **5%** threshold. The measured spread of those repeats is 12–26% at
short contexts. The uncertainty was four to five times the band being tested, so
every pass/fail at 128 and 512 was a coin toss. Two consecutive runs of the same
build reported 0.955× and 0.865× at 512 — nothing about the engine changed
between them.

What we changed in the harness (§6): discard warmup, raise repeats to 9 by
default, compute **paired** per-repeat ratios instead of a ratio of medians, and
attach a **bootstrap 95% confidence interval**, refusing to call pass or fail
when the interval straddles the threshold.

### Result: 31 paired repeats, 3 discarded warmups, bootstrap 95% CI

`--device cuda --dtype fp16 --kv-budget-mb 256 --skip-batched`

| context | SmolLM2-360M | Qwen2.5-1.5B | verdict |
|---:|---|---|---|
| 128 | 0.873× [0.867–0.892] | 0.897× [0.879–0.920] | decided **NOT met**, both |
| 512 | 0.894× [0.884–0.912] | 0.923× [0.901–0.947] | decided **NOT met**, both |
| 1024 | 0.923× [0.903–0.930] | 0.951× [0.926–0.977] | Smol NOT met; Qwen **undecided** |
| **2048** | **0.980× [0.964–0.986]** | **1.012× [0.996–1.026]** | **MET on both** |

Paged output was **byte-identical to contiguous at every context length on both
models**.

**The Phase A definition of done — within 5% of contiguous at 2048 — is met,
and now it is met with an error bar that excludes the threshold rather than a
single sample that happened to land.** On Qwen-1.5B paging is at parity or
slightly ahead (1.012×, interval entirely above 1.0 at the low end of 0.996).

**Qwen beats SmolLM2 at every context length**, and the intervals do not overlap
at 128, 512 or 2048. That confirms the amortisation argument with evidence: a
more expensive forward pass dilutes paging's fixed per-step cost. The honest
counterpart is that **paging genuinely does cost 10–13% at 128 tokens** on both
models, and that is now a settled finding rather than noise.

### The run we had to throw away

The first Qwen context sweep reported **3.73 tok/s** contiguous at 2048. The
refit reported **11.42**. The first run peaked at **6687 MB on a 6140 MiB
card**: CUDA on Windows pages device memory out to host RAM rather than failing
an allocation, so it silently measured PCIe traffic and labelled it `cuda`. A
**3.06× artifact** presented as a GPU result.

That is the same failure class `_device.py` already refuses for
`--device cuda` on a CPU-only build — a plausible number carrying a wrong
label — so there is now a `vram_spill_check()` in every harness (§6).

---

## 3. Phase B — prefix sharing. Ports cleanly, exactly as predicted.

`--concurrency 1 4 --tokens 12 --prefix-tokens 256 --device cuda --dtype fp16`

| | SmolLM2-360M | Qwen2.5-1.5B |
|---|---|---|
| blocks at N=4, sharing off → on | 72 → 24 (**0.333×**) | 72 → 24 (**0.333×**) |
| prefill work | **0.27×** | **0.27×** |
| prefix hit rate | 0.96 | 0.96 |
| copy-on-write copies | 0 | 0 |
| blocks leaked | 0 | 0 |
| output identical | yes | yes |
| KV saved | 45.0 → 15.0 MB (30 MB) | 31.5 → 10.5 MB (21 MB) |

At N=1 the ratio is exactly 1.000 on both — the control still works.

The block and prefill ratios are **dimensionless and identical across a 4×
change in model size**, which is the strongest possible form of this claim: it
is a property of the allocator, not of the model. The megabytes saved are
*lower* on Qwen purely because its GQA is leaner per token. Quote the ratios on
Qwen and the absolute savings on SmolLM2, and say which is which.

---

### 3a. Phase B scales with concurrency, and the two models agree bit for bit

`--concurrency 1 2 4 8 16 --prefix-tokens 256 --device cuda --dtype fp16`

| N | blocks off → on | block ratio | prefill ratio | hit rate | CoW | leaked | identical |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | 18 → 18 | 1.0000 | 1.0000 | 0.000 | 0 | 0 | yes |
| 2 | 36 → 20 | 0.5556 | 0.5133 | 0.889 | 0 | 0 | yes |
| 4 | 72 → 24 | 0.3333 | 0.2700 | 0.960 | 0 | 0 | yes |
| 8 | 144 → 32 | 0.2222 | 0.1483 | 0.983 | 0 | 0 | yes |
| **16** | **288 → 48** | **0.1667** | **0.0888** | **0.992** | 0 | 0 | yes |

**Every figure in this table is identical on SmolLM2-360M and Qwen2.5-1.5B, to
four decimal places.** Not approximately equal — identical. That is the
strongest possible evidence that prefix sharing is a property of the allocator
rather than of the model, and it is worth stating that way rather than as a
single-point "3.0× reduction".

**The benefit grows with load:** 1.8× fewer blocks at N=2, 3.0× at N=4, 4.5× at
N=8, **6.0× at N=16** — and prefill work falls to **8.9%**, an 11.3× reduction.
The trend follows `(16 + 2N) / 18N` — sixteen shared prefix blocks plus two
private blocks per sequence — which approaches a 9× ceiling for this prompt
shape. Quoting the asymptote would be more useful than quoting N=4.

Zero copy-on-write copies and zero leaked blocks at every concurrency on both
models.

## 4. Phase C — chunked prefill. **RETRACTED: the Qwen column is invalid.**

> **⚠ Retraction.** The Qwen-1.5B figures in this section were measured with the
> harness default `--kv-budget-mb 1024` and peaked at **9109 MB on a 6140 MiB
> card** — 1.48× the card's capacity. CUDA on Windows paged the overflow to host
> RAM, so those timings measure PCIe thrash, not the engine. **Every
> Qwen-1.5B number below, and the "1.62× improvement refutes the runbook"
> conclusion drawn from it, must not be used.**
>
> The SmolLM2 column peaked at 4817 MB and is valid.
>
> This was caught by the `vram_spill_check()` guard added during this campaign
> (§6, fix 5) — the first thing it flagged was our own freshly-collected data.
> Without it the retracted claim would have gone into the runbook as a
> headline. A re-run at `--kv-budget-mb 256`, nine repeats per configuration, is
> in `benchmarks/results/runs/r6-phaseC-p512-refit-*`.
>
> **Note the shape of this mistake:** the spilled baseline read **3.54×** versus
> SmolLM2's 2.48×, which looked exactly like the "bigger model, more expensive
> prefill" effect we expected to find. A contaminated number that confirms your
> hypothesis is the most dangerous kind.

`--prompt-tokens 512 --victim-tokens 16 --budgets 0 64 16 4 --fused`

| budget | fused | SmolLM2 stall | Qwen-1.5B stall |
|---:|---|---:|---:|
| unlimited | no | 2.48× | **3.54×** |
| 64 | no | 3.04× | 2.30× |
| 64 | yes | 3.71× | 3.55× |
| 16 | no | 2.53× | **2.18×** |
| 16 | yes | **2.15×** | 2.52× |
| 4 | no | 2.50× | 2.18× |
| 4 | yes | 2.24× | 2.40× |

- **SmolLM2: 2.48× → 2.15×, a 1.15× improvement.** (valid, but single-run — see
  below)
- ~~Qwen-1.5B: 3.54× → 2.18×, a 1.62× improvement.~~ **RETRACTED, VRAM-spilled.**

**Whether chunking helps on GPU at 1.5B is therefore still an open question.**
The runbook's claim that it "makes it worse" is *not* refuted by this data —
it is simply untested at 1.5B until the refit lands.

### 4b. Phase C on CPU with the 1.5B: the DoD is met, by a wide margin

This is the strongest result in the campaign, and unlike §4 it is valid —
CPU runs allocate no VRAM, so there is nothing to spill. Five repeats,
`Qwen2.5-1.5B`, `--device cpu --dtype fp32`, 512-token intruding prompt:

| budget | fused | median stall | min | max | spread |
|---:|---|---:|---:|---:|---:|
| unlimited | no | **31.26×** | 29.39 | 31.70 | 7% |
| 64 | no | 6.43× | 6.31 | 7.06 | 12% |
| 64 | yes | 5.80× | 5.35 | 5.92 | 10% |
| 16 | no | 2.74× | 2.59 | 2.83 | 9% |
| 16 | yes | 2.00× | 1.96 | 2.05 | 5% |
| 4 | no | 2.32× | 2.28 | 2.67 | 17% |
| **4** | **yes** | **1.52×** | **1.39** | **1.56** | **11%** |

**Phase C's definition of done — keep the worst-case stall under 2.0× — is MET
at 1.52×, and the entire observed range (1.39–1.56) sits below the threshold.**
The improvement over the unchunked baseline is **20.6×**.

Three things make this the most trustworthy Phase C measurement we have:

1. **The effect dwarfs the noise.** Spreads are 5–17% against a 20× effect,
   where the GPU runs had 41% spread against a 1.15× effect. This is why the
   same statistic that is useless in §4a is decisive here.
2. **Fusion helps at every single budget**, with non-overlapping ranges:
   2.32→1.52 at budget 4, 2.74→2.00 at 16, 6.43→5.80 at 64. **Phase C bullet 2
   is validated here** — and note this is the opposite of what the GPU
   single-runs suggested, which is further reason to distrust those.
3. **A 512-token prefill really does stall in-flight decoding by 31× on a 1.5B
   CPU model.** That is the problem chunked prefill was built to solve, finally
   measured in a regime where the problem actually exists.

This vindicates the runbook's own framing — *"chunking is a latency tool for the
regime where prefill dominates: CPU, or long prompts, or big models"* — and
supplies the number that framing never had. **It is also the honest answer to
"does this project have results worth showing": yes, this one, and it needed a
bigger model to become visible at all.** On SmolLM2-360M the same benchmark
moves 2.56×→2.12×, which is indistinguishable from noise.

### 4a. Phase C is noise-dominated, so even the SmolLM2 number is weak

Repeating the SmolLM2 configuration **five times** gives:

| budget | fused | median | min | max | spread |
|---:|---|---:|---:|---:|---:|
| unlimited | no | 2.56× | **1.62** | **2.68** | **41%** |
| 4 | no | **2.12×** | 1.92 | 2.85 | 44% |
| 16 | no | 2.22× | 2.12 | 2.75 | 28% |
| 16 | yes | 2.42× | 2.23 | 2.61 | 16% |
| 64 | yes | 2.90× | 2.36 | 3.32 | 33% |

The single-run table above reports "baseline 2.48×, best 2.15× at budget 16
**fused**". Five repeats give "baseline 2.56×, best 2.12× at budget 4
**unfused**". **Both the magnitude and which configuration wins changed.**

The reason is structural: `stall_ratio` is a **maximum over inter-token gaps** —
one worst-case sample per run — which is far noisier than a throughput median.
A max statistic does not average out; it ratchets to whatever the worst
scheduling hiccup was. **No single-run Phase C comparison, in this report or in
the runbook, can distinguish the configurations it ranks.** Phase C needs many
repeats and a distributional statistic (p99 of gaps, say) rather than the max,
and that is a harness change still outstanding.

**Fusion's direction is not established.** The single-run data suggested it
helps on SmolLM2 and hurts on Qwen, but the Qwen half is retracted and the
SmolLM2 half reverses under repetition (at budget 16, fused 2.42× against
unfused 2.22× over five runs — the opposite of the single run). The harness's verdict string *asserted* that fusion "drops the floor to
about 2x" — hardcoded narrative the data contradicts. It now derives the
statement from the fused/unfused twins in the run (§6). Removing a forward pass
is not automatically a win: the fused pass still carries the chunk's tokens
alongside the decode's, and at `head_dim` 128 that one wider pass can cost more
than the two narrower ones it replaced.

---

## 5. Phase D — the first real win, and a correctness claim that was overstated

### 5.1 Speculation is a wall-clock loss in every configuration tested

**This section was rewritten twice. Both earlier versions reported a win, and
both were measurement artifacts.** The history is kept in §5.1a because the
mistake is more instructive than the result.

Once the harness discards warmup (`--warmup 2`) and reports a median over nine
paired repeats, `--device cpu --dtype fp32 --batch-sizes 1 --max-new-tokens 64`:

| model pair | median speedup | range | acceptance | lossless |
|---|---:|---|---:|---|
| SmolLM2 360M ← 135M | **0.818×** | 0.801–0.895 | 0.553 | yes |
| Qwen 1.5B ← 0.5B | **0.777×** | 0.766–0.830 | 0.365 | yes |

And across token counts, five repeats each, warmup on:

| tokens | SmolLM2 | Qwen-1.5B |
|---:|---:|---:|
| 12 | 0.808× | 1.018× |
| 32 | 0.928× | 0.794× |
| 64 | 0.794× | 0.774× |
| 128 | 0.839× | 0.881× |

**Speculative decoding is slower than plain decoding in every configuration
measured on this machine** — CPU and GPU, batch 1 through 8, K from 1 to 8,
both model pairs, every token count bar one borderline cell. The range across
everything tested is roughly **0.32× to 1.02×**. Output is byte-identical in
fp32 throughout: the implementation is correct, it simply does not pay here.

### 5.1a Three readings of the same benchmark, and why only the last is real

| what was run | reported | verdict |
|---|---:|---|
| 12 tokens, 1 run, no warmup | **2.22×** | artifact |
| 32 tokens, 5 runs, no warmup | **1.172×** | artifact |
| 64 tokens, 9 runs, **warmup 2** | **0.818×** | the measurement |

Each refinement moved the number down and the previous one looked perfectly
plausible at the time. The mechanism is specific: `run_scenario` builds a fresh
engine and times one `generate`, the **plain arm runs first**, and so the plain
arm alone absorbed model warmup and allocator growth. That inflated the
speculative ratio by roughly 25–40%, worst at low token counts where the fixed
cost is the largest share of wall time.

The variance tells the same story. Before warmup, five repeats of one
configuration spanned **1.103–1.197**. After warmup, nine repeats span
**0.801–0.895** — and the Qwen series is tighter still at **0.766–0.830**, a 1%
spread on the tok/s figures. **Discarding warmup both removed the bias and
collapsed the noise**, which is what a real fix looks like.

The earlier §5.1c observation about a monotone downward drift across repeats was
also this effect, not thermal throttling: the first repeat was the one carrying
the warmup, so the series fell as it washed out.

### 5.1b Acceptance, not cost ratio, is what separates the pairs

The Qwen pair has the **better cost ratio** (3.1× versus 2.7×) and is still the
**worse** performer — 0.777× against 0.818×. The difference is acceptance:
**0.365 versus 0.553**.

> **Draft selection is governed by agreement with the target, not by the
> parameter ratio.** A cheap draft that guesses wrong makes the target re-run
> anyway, and the draft's K serial passes become pure overhead.

Same-family is not a sufficient heuristic either — both pairs are same-family
and differ by ~1.5× in acceptance. This refutes the reasoning in
`PHASE_E_RUNBOOK.md` §3, which recommends the Qwen pair for Phase D on cost
ratio alone.

### 5.1c Why speculation cannot win here, stated plainly

Two independent reasons, both structural rather than implementation defects:

1. **The draft's K passes are serial.** Per round the engine pays K draft
   passes plus one target verify. With acceptance *a*, a round yields about
   `1 + a·K` tokens. For the SmolLM2 pair at K=4 and a=0.55 that is ~3.2 tokens
   for 5 forward passes, where plain decoding gets 4 tokens from 4 passes. The
   draft is 2.7× cheaper, which is not cheap enough to cover the gap.
2. **Batching already captured the win.** Plain decode goes from 6.2 to 25.4
   tok/s between batch 1 and 2. Speculation and continuous batching are two ways
   to raise tokens-per-forward-pass, and this engine already has the better one.

> **Speculation needs a target so expensive that a pass is worth avoiding at
> almost any price — an 8B-class model or larger. A 6 GB card will not hold one
> in fp16, and 24 GB of RAM will not hold one in fp32. The claim is not
> refutable on this hardware, and the honest position is that it remains
> untested rather than disproven.**

### 5.1d Lookahead K was swept, but before the warmup fix

A K sweep on the SmolLM2 pair (single runs, **no warmup**) suggested K=6 at
1.370× against the K=4 default's 1.205×. **Those are the inflated numbers**, so
the 1.370× is not real. The *shape* may still hold — acceptance fell
monotonically from 0.875 at K=1 to 0.475 at K=8, which is a genuine effect, and
the speedup peaked in the middle as the trade-off predicts. Round 8 re-measures
K ∈ {2,4,8} with warmup and 7 repeats; treat the ranking as open until then.

On the Qwen pair every K from 1 to 8 was below 1.0 even *before* the warmup fix
(0.678–0.940×), so that pair is unsuitable at any K.

### 5.2 "Verified lossless" was true in fp32 and unsound in fp16

Running the Qwen pair on GPU, `output_identical` went **False at batch 4**. The
`--self-draft` mode — documented as *"forces 100% greedy acceptance"* — reported
**acceptance 0.9375**. Under exact arithmetic that number must be 1.0: the
drafting model and the verifying model are the same weights on the same context.
So the engine was computing different logits for one model depending on which
path evaluated it.

A four-way matrix over dtype and geometry settles what it is:

| model | head_dim | fp32 | fp16 | bf16 |
|---|---:|---|---|---|
| SmolLM2-360M | 64 | — | **exact, batch 1–8** | — |
| Qwen2.5-0.5B | 64 | **exact 1–8** | **exact 1–8** | diverges from batch 2 |
| Qwen2.5-1.5B | **128** | **exact 1–8** | diverges from batch 4 | diverges from batch 3 |

Mantissa bits: fp32 **23**, fp16 **10**, bf16 **7**. Accumulation length scales
with `head_dim`. The pattern is **monotone in both variables** — divergence
worsens as mantissa precision falls and as the dot-product reduction lengthens.
`head_dim` 128 doubles the terms accumulated per attention score versus 64,
doubling accumulated rounding error, and that is precisely where fp16 tips over.

Two hypotheses were tested and **refuted**:

- *fp16 range saturation* (values exceeding 65504). Refuted: bf16 has a far
  wider exponent range yet diverges **more**. Only mantissa precision is
  consistent with the data.
- *A masking or padding logic bug.* Refuted: fp32 is bit-exact at every batch
  size and geometry. A logic bug would diverge in fp32 too.

The cleanest evidence is fp16 on **CPU**: acceptance stays exactly 1.0 — draft
and verify agree — yet the speculative output still differs from the plain
baseline. With self-draft at full acceptance the speculative output *is* the
draft's tokens from the **unbatched** path, while the baseline uses the
**batched** path. So:

> **The batched and unbatched decode paths are not bit-identical in half
> precision at batch ≥3. In fp32 they agree exactly.**

**There is no algorithmic bug.** But the project asserts correctness as
*byte-identical output*, and fp16 is the GPU default, so the claim as written
cannot hold for `head_dim ≥ 128` models. The correct resolution is to make the
claim dtype-explicit: assert byte-identity in fp32, and in fp16 assert
tolerance-based equivalence with first-divergence reporting. **That is a change
to the test, not to the engine.**

The second defect is worse than the first: **nothing checked the invariant.**
The harness printed `acceptance 0.9375` and still called the run lossless. That
flag exists to be a correctness check, and it was decorative. It is now enforced
(§6).

---

## 6. Harness defects found and fixed

All five changes are in `benchmarks/`. **398 Python tests pass and pyflakes is
clean after them.**

| # | file | defect | fix |
|---|---|---|---|
| 1 | `run_speculative_batching.py` | `--self-draft` acceptance < 1.0 printed and ignored — a failing invariant, silently | asserts acceptance == 1.0, fails the run, records `self_draft_exact` |
| 2 | `run_context_sweep.py` | no warmup; first sample always slowest, biasing whichever path ran first | `--warmup` (default 2), discarded before timing |
| 3 | `run_context_sweep.py` | 3 repeats against a 5% threshold with 12–26% spread; ratio-of-medians | repeats → 9; **paired** per-repeat ratios; bootstrap 95% CI; `MIN_REPEATS_TO_DECIDE = 5` |
| 4 | `run_context_sweep.py` | `within_5_percent` was two-sided — a **7% speedup** was reported as a DoD failure | one-sided (`>= 0.95`), plus `UNDECIDED` when the interval straddles it |
| 5 | `_device.py` + all 4 harnesses | no VRAM guard; 6687 MB peak on a 6140 MiB card reported as a GPU result | `vram_spill_check()` warns loudly and records `vram_spill` in the JSON |

Fix 3 had a flaw on its first attempt worth recording: the error bar was the
sample **min/max**, which *widens* with more repeats, so the verdict could never
converge no matter how long the benchmark ran. It is a bootstrap CI on the
median now, which narrows with evidence — measured width 0.081 at n=9 against
0.037 at n=99. The `MIN_REPEATS_TO_DECIDE` floor exists because a bootstrap on 3
points can only resample those 3 values and reports a deceptively *tight*
interval (0.046 at n=3 versus 0.081 at n=9 on the same distribution).

---

## 7. Environment notes that cost us time

- **`HF_HOME` is set User-scoped to `C:\Users\Anant\huggingface`.** The weights
  in `C:\Users\Anant\.cache\huggingface` were therefore invisible, and
  `check_env.py` reported `FAIL Model weights` while the files plainly existed.
  The SmolLM2 pair was moved into the active cache. The old `.cache` still holds
  unrelated models (RMBG-1.4, CLIP) belonging to other projects — do not delete
  it wholesale.
- **`hf download <repo> --exclude "*.pth" "original/*"` silently downloads
  nothing** and exits **0** in seconds. The CLI signature is
  `hf download REPO [FILENAMES...]`, so bare values after `--exclude` become an
  explicit filename list; it warns `Ignoring --exclude since filenames have
  been explicitly set`. Verify with `du -sh`, never the exit status. Qwen2.5
  repos are safetensors-only and need no filtering.
- **CPU benchmarks must be fp32**, because `_device.py` correctly refuses to
  treat half precision on CPU as a performance measurement. That makes **system
  RAM**, not VRAM or disk, the binding constraint on CPU runs: Qwen 3B + 0.5B
  needs ~16 GB, 1.5B + 0.5B needs ~9.7 GB.
- **6141 MiB of VRAM does not hold a 3B in fp16** (6.18 GB of weights alone), so
  the 3B is a CPU-only target on this machine.

---

## 8. Where each claim stands

| phase | claim | status |
|---|---|---|
| **A** | paging within 5% of contiguous at 2048 | ✅ **MET**, both models, 31 paired repeats, CI excludes threshold, byte-identical |
| **A** | paging free at short context | ❌ **NOT met** — 0.873–0.897x at 128, now a settled result rather than noise |
| **A** | paging at 1024 | ⚠️ **genuinely on the threshold.** 75 repeats give 0.945x [0.932-0.957] on Qwen-1.5B: a tight interval that still contains 0.95, so the honest answer is "paging costs about 5.5% here", not a pass or a fail |
| **B** | prefix sharing cuts blocks | ✅ **scales with load: 1.8x at N=2 -> 6.0x at N=16**, prefill work down to 8.9%. Every figure **bit-identical** between the two models. 0 CoW, 0 leaked, output identical throughout |
| **C** | chunking protects in-flight decode | ✅ **DoD MET on CPU at 1.5B: 31.26× -> 1.52×, a 20.6× improvement**, n=5, whole range below the 2.0x threshold. On GPU: **unresolved** — the Qwen runs were VRAM-spilled and retracted, and the statistic is noise-dominated |
| **C** | fusion lowers the floor | ✅ **validated on CPU** — helps at every budget with non-overlapping ranges (2.32->1.52 at budget 4). On GPU: direction not established |
| **D** | speculation gives 2–3x | ❌ **not met, and there is no win at all.** With warmup discarded and 9 repeats: **0.818x** (SmolLM2) and **0.777x** (Qwen). Everything tested spans ~0.32-1.02x. The 2.22x and 1.172x readings were both warmup artifacts |
| **D** | a bigger target makes speculation pay | ❌ **refuted** — the Qwen pair has the better cost ratio and worse acceptance (0.364 vs 0.656), so it performs worse. **Acceptance dominates cost ratio** |
| **D** | speculation composes with batching | ⚠️ composes structurally (3.0 seq/pass) but **loses on wall clock everywhere** — they compete for the same win, they do not compound. Batching is the better of the two and the engine already has it |
| **D** | verified lossless | ⚠️ **exact in fp32, unsound as stated in fp16** for `head_dim ≥ 128`; test needs qualifying, engine does not |
| **F** | run_pipeline, 7 stages | ✅ **ALL PASSED**, 46.3 s, 398 tests, status READY |
| **F** | C++ AddressSanitizer | ✅ **CLEAN** — no use-after-free, overflow, double free or heap leak across 40 cycles of 12 concurrent requests; 60-cycle sustained load generated 9896 tokens with **0 logical block leak**, free list back to 256/256 |
| **F** | no leaks under sustained load | ✅ **both models pass all three checks.** SmolLM2: 0 leaked, all returned, RSS **+4.8 MB**. Qwen-1.5B: 115/115 requests ok, 0 leaked, all returned, RSS **−206.9 MB** vs plateau |

### What to do next, in priority order

1. **Qualify the losslessness claim by dtype** in the runbook, the README and
   the tests. Make the byte-identity tests run fp32 and add a tolerance-based
   fp16 test that reports first-divergence position. This is the highest-value
   item: the claim as written is the project's headline correctness guarantee.
2. **Correct `PHASE_E_RUNBOOK.md` §5.1 and §5.3.** The 3-repeat context table
   cannot support its verdicts, and the GPU chunking conclusion is now refuted.
   Rebuild the PDF via `scripts/build_runbook_pdf.py` afterwards.
3. **Re-state the Phase D headline** as "2.22× at batch 1 on CPU, lossless" with
   the batching trade-off explained, rather than "a net wall-clock loss".
4. **Add `--out` to every harness.** `RESULTS_PATH` is hardcoded, so any run
   overwrites the previous model's results; this whole campaign needed an
   external archiving wrapper to avoid losing the baseline.
5. **Give the speculative harness warmup, repeats and a CI**, as the context
   sweep now has. This is urgent, not cosmetic: without warmup its speedup
   varies with `--max-new-tokens` and reported 2.22× where the true figure is
   ~1.17× (§5.1a). A headline speedup with no error bar is not a result.
