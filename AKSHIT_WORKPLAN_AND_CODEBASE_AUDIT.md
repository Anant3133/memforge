# MemForge — Track 2 (Akshit) & Codebase Hygiene Audit

**Part A:** Track 2 — Phase A performance work, with a measured architecture
**Part B:** Codebase audit — dead code, hardcoded values, stubs
**Written:** 2026-10-25 · **Author:** Anant (Track 3 lead, integration owner)
**Companion docs:** [TEAM_EXECUTION_PLAN.md](TEAM_EXECUTION_PLAN.md) · [ANANT_WORKPLAN.md](ANANT_WORKPLAN.md) · [ASHUTOSH_SANSKAR_BHUMIKA_WORKPLAN.md](ASHUTOSH_SANSKAR_BHUMIKA_WORKPLAN.md) · [PROGRESS_AND_NEXT_PHASES.md](PROGRESS_AND_NEXT_PHASES.md)

> Every performance number in Part A was **measured on this machine today**, not
> estimated. Two of the measurements contradict the original plan, and §A.2
> states that plainly rather than quietly following the plan anyway.

---

# Part A — Track 2: the paged-attention read path

## A.1 What the plan asks for

From [TEAM_EXECUTION_PLAN.md](TEAM_EXECUTION_PLAN.md) §3.1 and
[PROGRESS_AND_NEXT_PHASES.md](PROGRESS_AND_NEXT_PHASES.md) §8 Phase A:

| # | Deliverable | Plan's stated approach |
| --: | :--- | :--- |
| 1 | `memforge_ml/kernels/paged_gather.py` | "Replace the per-step Python list comprehension / nested indexing with a vectorized `torch.index_select` or preallocated destination buffer" |
| 2 | In-place attention read | "Adapt `PagedCache` to allow attention modules to gather directly without temporary contiguous cache round-trips" |
| 3 | `benchmarks/run_context_sweep.py` | Throughput across 128 / 512 / 1024 / 2048 tokens |
| | **Definition of done** | "Paged throughput within 5% of contiguous at 2,048 tokens" |

The motivating limitation, from §7.1 of the progress report: *"Reading a
sequence's KV costs `O(context_length)` indexing per layer per step… the gather
is `O(L)` where a fused kernel would be `O(1)` pointer arithmetic. This is the
single biggest performance item remaining."*

## A.2 What measurement says — including where the plan is wrong

Before designing anything I measured the actual hot path: SmolLM2-360M geometry
(32 layers, 5 KV heads, head_dim 64, fp32), a real 320 MB pool, block size 16.

### A.2.1 The gather really is O(context), as documented

Cost of one decode step's KV read — all 32 layers, K and V, against the live
`PagedKVPool`:

| Context | Full gather | Index only | `permute().contiguous()` tax |
| ---: | ---: | ---: | ---: |
| 128 | 22.6 ms | 8.5 ms | 6.2 ms |
| 512 | 26.7 ms | 15.2 ms | 8.7 ms |
| 1024 | 41.1 ms | 22.8 ms | 15.2 ms |
| 2048 | **79.4 ms** | 45.5 ms | 25.5 ms |

Linear in context, as the limitation claims. At 2048 tokens the engine spends
**79 ms per decode step just reading KV**, before any attention arithmetic. A
third of it (25.5 ms) is the `permute(1,0,2).unsqueeze(0).contiguous()` in
[kv_pool.py:136-143](memforge_ml/runtime/kv_pool.py#L136-L143) — pure layout
tax, because the pool stores `(T, H, D)` and attention wants `(1, H, T, D)`.

### A.2.2 The plan's suggested fix does not work

The plan says to use "a vectorized `torch.index_select` or preallocated
destination buffer". Both were prototyped and measured against the current code:

| Context | Current | Flattened `index_select` | `index_select` into a preallocated buffer |
| ---: | ---: | ---: | ---: |
| 128 | 1.41 ms | 0.96 ms (1.5×) | 1.21 ms (1.16×) |
| 512 | 2.90 ms | 2.37 ms (1.2×) | 2.99 ms (**0.97×**) |
| 1024 | 4.27 ms | 3.43 ms (1.2×) | 4.40 ms (**0.97×**) |
| 2048 | 7.64 ms | 6.37 ms (1.2×) | 8.22 ms (**0.93×**) |

*(Smaller absolute figures than A.2.1 because this isolates K only on a
cache-resident tensor; the ratios are what matter.)*

**The preallocated destination buffer is slower than what we have.** Writing
through a transposed view (`dest.transpose(0,1).copy_(src)`) has a worse memory
access pattern than `permute().contiguous()`, which dispatches to an optimised
copy kernel. Verified byte-identical output, so this is a performance result and
not a correctness artefact.

Flattened `index_select` gives a real but modest **1.2×**. Worth taking, nowhere
near the DoD.

**Conclusion: the plan's Phase A approach, implemented exactly as written,
would deliver roughly 1.2× and miss the 5%-of-contiguous target by a wide
margin.** The work needs a different shape, which is §A.3.

### A.2.3 The actual win is algorithmic, and it is 10×

The engine re-gathers **the entire context on every decode step**, even though
exactly one token was added since the last one. Maintaining a persistent
per-layer buffer and appending only the new token's K/V:

| Context | Rebuild everything (today) | Append only | Speedup |
| ---: | ---: | ---: | ---: |
| 128 | 2.87 ms | 0.99 ms | 2.9× |
| 512 | 5.39 ms | 1.08 ms | 5.0× |
| 1024 | 9.92 ms | 1.15 ms | 8.6× |
| 2048 | **14.91 ms** | **1.46 ms** | **10.2×** |

The shape matters more than the ratio: append-only is **nearly flat in context**
(0.99 → 1.46 ms across a 16× context increase) where rebuilding is linear. That
is the `O(L)` → `O(1)` change the limitation actually asks for.

### A.2.4 The biggest cost is not the gather at all

This is the finding that most changes the plan. Limitation 2 — the batched
decode round-trip through a temporary contiguous cache
([continuous_engine.py:503-519](memforge_ml/runtime/continuous_engine.py#L503-L519))
— gathers every sequence, left-pads each with `F.pad`, and `torch.cat`s them per
layer. Measured, pad + cat alone:

| Batch | Context | pad + cat per step | Bytes copied per step |
| ---: | ---: | ---: | ---: |
| 4 | 512 | 1.4 ms | 80 MB |
| 8 | 512 | 3.5 ms | 160 MB |
| 8 | 1024 | 12.9 ms | 320 MB |
| 8 | 2048 | **46.8 ms** | **640 MB** |

**At batch 8 / context 2048 the engine copies 640 MB per decode step and spends
47 ms doing it — roughly 3× the gather it sits on top of.** The plan ranks the
fused gather first and the in-place read second; the measurement says the
ordering should be reversed. §A.4 sequences it accordingly.

### A.2.5 One thing not worth optimising

Index translation (`slot_indices`: `arange` + `div` + `mod` + a table lookup)
costs **0.049 ms at 2048 tokens** — 0.06% of the 79 ms gather. An append-only
version is 2.2× faster and saves 0.027 ms. Recorded so nobody spends a day on
it: it is already free, and the existing per-step memoisation across layers is
sufficient.

## A.3 Architecture

Three tiers, ordered by value per unit of risk. Each is independently shippable
and independently measurable, which matters because tier 3 is research and tiers
1-2 are not.

```mermaid
graph TB
    subgraph Model["Hugging Face model (unmodified)"]
        Attn["attention layer<br/>wants (1, H, T, D)"]
    end

    subgraph T1["Tier 1 — remove redundant copies (no memory cost)"]
        Flat["flattened index_select<br/>1.2x"]
        NoPad["batched decode without pad+cat<br/>saves 640 MB/step at B=8, ctx=2048"]
    end

    subgraph T2["Tier 2 — append-only materialisation (10x, costs memory)"]
        Append["persistent per-sequence view buffer<br/>O(1) per step instead of O(context)"]
        Evict["released on finish / preempt / abort"]
    end

    subgraph T3["Tier 3 — block-wise attention (the real fix, no memory cost)"]
        Online["online-softmax attention over blocks<br/>never materialises the context"]
        Triton["GPU: Triton kernel (Phase E)"]
    end

    Pool["PagedKVPool<br/>(layers, blocks, slots, kv_heads, head_dim)"]

    Pool --> Flat --> Attn
    Pool --> Append --> Attn
    Pool --> Online --> Attn
    NoPad --> Attn
```

### A.3.1 Tier 1 — stop copying what does not need copying

**No memory cost, no behaviour change, byte-identical output.** Two pieces:

**(a) Flattened `index_select` in the gather.** The pool is
`(L, B, S, H, D)`; `pool[l].view(B*S, H, D)` is a free view, and
`physical * block_size + slot` is a single flat index. One `index_select`
replaces the two-dimensional advanced index. Measured **1.2×**, consistently.

**(b) Batched decode without the pad + cat.** This is the big one. Today's path:

```
for each layer:                      # 32 times
    for each sequence in the batch:  # up to 8
        gather its whole context     # O(ctx) copy
        F.pad to the batch maximum   # another O(maxlen) copy
    torch.cat                        # a third O(batch × maxlen) copy
build a throwaway DynamicCache from the result
```

Three full copies of the batch's entire KV per layer per step. The fix is to
gather **directly into one preallocated `(batch, H, maxlen, D)` buffer** at the
correct offset per sequence, so padding is "the region we did not write" rather
than a copy, and `cat` disappears because the destination was always one tensor.

The buffer is allocated once per `(batch, maxlen)` shape and reused; a shape
cache keyed on `(batch, maxlen_bucket)` avoids reallocating as contexts grow,
with `maxlen` rounded up to a block multiple so the bucket count stays small.

Expected: removes ~2 of the 3 copies, so roughly **2-3× on the batched decode
path** at batch 8. To be measured, not assumed.

### A.3.2 Tier 2 — append-only materialisation, with its cost stated

The 10.2× from §A.2.3, and the honest trade that comes with it.

A persistent per-sequence `(1, H, T_max, D)` buffer per layer is **a contiguous
copy of that sequence's KV**. Holding one per running sequence costs exactly
what contiguous caching costs — which is the thing paging exists to avoid. So
this cannot be a blanket change; it has to be scoped:

- **Only for sequences in the running set.** Allocated on admission, released on
  finish, preempt or abort — the same lifecycle as the block table, hooked into
  the same places (`_finish`, the preemption branch, `_process_pending_aborts`).
- **Bounded by `max_num_seqs × max_model_len`**, which is a number the operator
  already sets, so the worst case is predictable rather than emergent.
- **Opt-in** via `MEMFORGE_DECODE_VIEW_CACHE` (default off until measured on
  real weights), exactly as prefix sharing is opt-out via a constructor flag.
- **It does not undo prefix sharing.** Sharing saves blocks in the *pool*, which
  is sized by `kv_budget_mb` and is the constrained resource; the view buffer is
  transient working memory for the active batch. The two measure different
  things and §A.5 reports both so the trade is visible rather than implied.

**Correctness obligation:** the buffer is a cache of the pool, and the pool is
the source of truth. Any event that changes the logical→physical mapping
underneath it must invalidate it — copy-on-write (`prepare_write` repoints a
block), preemption (blocks freed), and a prefix-cache hit on admission (the
first `C` tokens come from someone else's blocks). Each of those is an existing
hook, and each needs an explicit invalidation plus a test.

### A.3.3 Tier 3 — block-wise attention, the actual fused kernel

Tiers 1 and 2 make materialising cheaper. Tier 3 stops materialising.

Attention over a paged cache can be computed block by block with an online
softmax (the FlashAttention formulation): for each 16-token block, compute
`q · kᵀ`, update a running max and sum, accumulate `p · v`. Working memory is
`O(block_size)` instead of `O(context)`, and the pool is read in place with no
intermediate tensor at all.

This is what "fused kernel" means, and it is also where the GPU path lives: the
same formulation becomes a Triton kernel in Phase E, which is why the structure
matters more than the CPU performance. It requires replacing the model's
attention module rather than just its cache, which is a real architectural step
and is why it is tier 3 rather than tier 1.

**Scoped as research, not as a deliverable with a date.** A correct, slow
reference implementation that proves equivalence is worth more than a fast one
that is subtly wrong, and the equivalence test is the real artefact. On CPU in
Python it may well be *slower* than tier 1+2; that result should be published
either way, in the manner of §3.5 of the progress report.

## A.4 Deliverables, resequenced by the measurements

| Order | File | What | Why this order |
| --: | :--- | :--- | :--- |
| 1 | `benchmarks/run_context_sweep.py` | Throughput and latency at 128/512/1024/2048, paged vs contiguous | **The benchmark comes first.** The DoD is a measurement, so without it nothing can be claimed, and every tier below needs it as the before/after |
| 2 | `memforge_ml/kernels/__init__.py`, `memforge_ml/kernels/paged_gather.py` | Tier 1(a): flattened `index_select` gather, behind a stable function boundary | Smallest, safest, 1.2×, and establishes the seam §A.6 requires |
| 3 | `memforge_ml/kernels/batched_decode.py` | Tier 1(b): gather directly into one preallocated batch buffer, no `F.pad`, no `cat` | **Largest measured cost** (47 ms, 640 MB per step at batch 8 / ctx 2048) |
| 4 | `memforge_ml/runtime/decode_view_cache.py` | Tier 2: append-only per-sequence buffer with invalidation hooks | The 10.2×, and the one with a memory cost to defend |
| 5 | `memforge_ml/kernels/block_attention.py` | Tier 3: reference online-softmax attention over blocks, with an equivalence test | Research; the equivalence proof is the deliverable |

The plan lists the fused gather first and the in-place read second. Items 2 and 3
invert that, because the in-place read is where the time actually goes.

## A.4b Measured results of tiers 1a and 1b (implemented 2026-10-25)

Both shipped. Output byte-identical throughout. The results are not what §A.3
predicted, in both directions.

### The Phase A definition of done was already met, before any optimisation

Baseline sweep (`run_context_sweep.py`, SmolLM2-360M, fp32, CPU, medians of 3
interleaved repeats), taken **before** either kernel landed:

| Context | Contiguous tok/s | Paged tok/s | Relative | Within 5%? |
| ---: | ---: | ---: | ---: | :--- |
| 128 | 13.50 | 12.42 | 0.920 | no |
| 512 | 12.49 | 12.02 | 0.962 | yes |
| 1024 | 10.38 | 10.74 | 1.034 | yes |
| 2048 | 8.23 | 7.99 | **0.971** | **yes** |

**"Paged throughput within 5% of contiguous at 2,048 tokens" was already true.**
It had simply never been measured at 2048 — the existing comparison in
`run_live_engine.py` is taken at roughly 40 tokens of context.

**Why the criterion is weak, which matters more than the result.** The baseline
it compares against is `DynamicCache`, which *also* concatenates its whole
context on every decode step. Both paths are `O(context)`:

| Context | Contiguous slowdown vs 128 | Paged slowdown vs 128 |
| ---: | ---: | ---: |
| 512 | 1.08x | 1.03x |
| 1024 | 1.30x | 1.16x |
| 2048 | **1.64x** | **1.55x** |

The ratio stays near 1.0 because both degrade together. So the definition of
done measures paged-vs-contiguous *parity*, not absolute efficiency, and is
structurally insensitive to the `O(L)` to `O(1)` problem that limitation §7.1
describes. A target that cannot fail for the reason it was written is worth
saying out loud rather than banking.

### Tier 1a — correct, and no measurable end-to-end effect

`memforge_ml/kernels/paged_gather.py`, wired into `PagedCacheLayer.update`.
In isolation the flattened `index_select` is the predicted ~1.2x faster. End to
end, across the same sweep:

| Context | Paged before | Paged after | Delta |
| ---: | ---: | ---: | ---: |
| 128 | 12.42 | 12.49 | 1.006x |
| 512 | 12.02 | 11.33 | 0.943x |
| 1024 | 10.74 | 10.44 | 0.972x |
| 2048 | 7.99 | 8.35 | 1.045x |

**That is noise, not a speedup.** The gather is not a large enough share of a
single-sequence decode step on this model for a 1.2x on it to surface above
run-to-run variance.

Kept regardless: it is byte-identical, no slower, simpler than the
two-dimensional advanced index it replaced, and it is the seam a Triton kernel
would substitute into in Phase E. But it is not a performance win and is not
claimed as one.

### Tier 1b — a real 1.3x, where §A.3.1 predicted 2-3x

`memforge_ml/kernels/batched_decode.py`, replacing gather → `F.pad` →
`torch.cat` with a gather straight into one preallocated, left-padded buffer.
Measured against the exact code it replaced, same pool, same indices:

| Batch | Context | Old pad+cat | New preallocated | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 512 | 39.3 ms | 37.0 ms | 1.06x |
| 8 | 512 | 88.3 ms | 68.7 ms | **1.28x** |
| 8 | 1024 | 143.8 ms | 105.5 ms | **1.36x** |
| 8 | 2048 | 293.2 ms | 221.5 ms | **1.32x** |

Consistent and real, and it removes the 640 MB-per-step copy that motivated it.
But **§A.3.1's estimate of "roughly 2-3x" was too optimistic and is wrong.**
The reasoning there — "removes 2 of the 3 copies" — counted copies without
weighting them. The copy that remains is the gather itself, a scattered read
that costs far more per byte than the two sequential copies removed. My
estimate, corrected by measurement.

### What this leaves

Tier 1 is done and worth about 1.3x on the batched decode path. The headline
`O(L)` to `O(1)` win is still entirely in **tier 2** (§A.3.2, measured at 10.2x
on the gather in §A.2.3), which has a memory cost to defend, and the
architecturally correct fix is still **tier 3** (§A.3.3). Neither is implemented.

## A.5 Definition of done

**Non-negotiable, for every tier:**

- [ ] Output **byte-identical** to the current engine. Every optimisation here
      changes how KV is read; a wrong read produces fluent, confident, wrong
      text rather than a crash, so the bar is exact token equality, as it is for
      prefix sharing
- [ ] All existing tests green (**157** Python, 3/3 C++)
- [ ] `python scripts/run_pipeline.py` 7/7

**Phase A target:** paged throughput within **5%** of contiguous at 2048 tokens,
reported by `run_context_sweep.py`.

> **On whether that target is reachable:** tier 1 + tier 2 remove the dominant
> per-step costs, and tier 2's append-only path is nearly flat in context, which
> is the property the 5%-at-2048 criterion is really testing. But the comparison
> baseline (`DynamicCache`) is *also* append-only and does no index translation
> at all, so parity is the ceiling, not a floor. If the result lands at, say,
> 12% rather than 5%, that gets reported with the residual costs itemised —
> exactly as the 1/5.7 prefix-sharing result was reported against a 1/8 goal.

**Must also be reported, not just throughput:**

- Peak resident KV with the tier 2 buffer enabled vs disabled, so the memory
  cost of the 10× is visible alongside it
- Block-accounting invariant (`total == free + allocated`) unchanged
- Prefix-sharing metrics unchanged, proving tier 2 did not quietly break it

---

# Part B — Codebase hygiene audit

Scanned on 2026-10-25: every `.py`, `.cpp`, `.hpp`, `.ts`, `.tsx` and `.md` in
the repo, excluding `.venv`, `node_modules` and `build/`. Method: AST-level
extraction of every top-level definition, then a reference search across all
code files (documentation mentions deliberately do **not** count as usage), plus
module-level import-graph analysis and targeted greps for stubs, TODOs,
hardcoded paths, models and ports.

## B.1 Dead code — confirmed, with the verdict

### B.1.1 Modules with zero importers

| Module | Lines | Verdict |
| :--- | --: | :--- |
| `memforge_ml/config.py` | 27 | **Delete.** A dataclass `EngineConfig` plus `DEFAULT_CONFIG`, imported by nothing. It also carries stale hardcoded model names (`meta-llama/Meta-Llama-3-8B-Instruct`, `TinyLlama-1.1B`) that were never used by the running engine. `server/config.py` is the real, env-driven configuration |
| `server/queues/request_queue.py` | 72 | **Delete.** `GenerationRequest` and `AsyncRequestQueue`, imported by nothing — `EngineManager` owns per-request `asyncio.Queue`s directly. It is also actively broken: line 19 calls `asyncio.get_event_loop()` **at class-definition time** as a dataclass default, which is deprecated and raises on Python 3.12+ when no loop is running. It never fires only because nothing imports it |

### B.1.2 Dead definitions inside live modules

| Symbol | Location | Verdict |
| :--- | :--- | :--- |
| `AbortGuard` | `server/api/disconnect_handler.py:107` | **Delete.** I wrote it; nothing uses it. `run_cancellable` covers every path that needed it. Its only other mention is its own docstring |
| `fetchMetrics`, `parseMetrics` | `frontend/src/services/api.ts` | **Delete.** I added them for a `/metrics` panel that was never built. `BenchmarkView` drives off the WebSocket telemetry feed instead |
| `fetchBlockMap`, `fetchTelemetry`, `sendChatCompletion`, `EngineBusyError` | `frontend/src/services/api.ts` | **Keep, deliberately.** Pre-existing REST clients the dashboard does not currently call because it uses the WebSocket feed. They are the documented REST surface, exercised by the API guide, and the natural fallback when WebSockets are blocked. `EngineBusyError` is thrown by `sendChatCompletion`. Removing them would delete the REST client, not dead code — **but they must be referenced somewhere or they are decoration**, so §B.4 wires them into the `StatusBanner` health probe rather than deleting them |
| `Scheduler::get_num_finished` | `engine/include/memforge/scheduler.hpp:57` | **Delete.** Header, implementation and a pybind binding, with zero callers anywhere. It is also `O(n)` over every sequence ever submitted |
| `BlockAllocator::reset` | `block_allocator.hpp:129` | **Keep.** Zero callers today, but it is the documented reset hook for benchmarks and it is 8 lines. **Must gain a caller or go** — §B.4 uses it in the microbenchmark that currently rebuilds an allocator per iteration |
| `PhysicalBlock::last_accessed_timestamp` | `types.hpp:35` | **Delete.** Never written, never read, by anything |
| `CachedBlock::last_hit_timestamp`, `PrefixCache::clock_` | `prefix_cache.hpp:66,180` | **Delete.** Written on every hit and read by nothing. I added them for the stage-2 LRU eviction that §10.5 of the progress report explicitly defers, so they are speculative state today. Stage 2 can reintroduce them when it needs them |
| `SequenceStatus::SWAPPED` | `types.hpp:23` | **Keep, but fix the comment.** Never set: preemption is by recomputation, not swapping, so the comment "swapped to host memory" describes behaviour that does not exist and misleads anyone reading `types.hpp` first. Removing an enum value from a bound API is a gratuitous break; mislabelling it is the actual defect |

### B.1.3 `BlockAllocator::free_sequence` — not dead, but unsafe

Referenced only by `tests/cpp/test_block_allocator.cpp`. The scheduler stopped
using it when prefix sharing landed, in favour of `release_sequence_blocks()`.

It is worse than dead: **it frees a sequence's blocks without reporting which
reached `ref_count` 0, so the PrefixCache registry is never told.** A caller
with shared blocks would leave stale entries pointing at freed blocks — the
precise hazard the reverse `by_block_` index exists to prevent, reachable
through a public, Python-bound method.

**Verdict: keep the name, make it safe** — return the zero-ref ids like
`release()` does, so a caller physically cannot forget to evict. Deleting it
would be fine too, but a safe `free_sequence` is a reasonable API and the
unsafe one is a trap.

### B.1.4 C++ `EngineConfig`

`types.hpp:77` defines it and `bindings.cpp:50-55` exposes it; **nothing in the
Python runtime constructs or reads it.** `num_gpu_blocks: 1024` is the most
misleading part — the pool is sized from `kv_budget_mb`, so that field has never
affected anything (already recorded in [ANANT_WORKPLAN.md](ANANT_WORKPLAN.md)
§0.4.3).

**Verdict: delete the struct, its binding and its stub.** It is the same
situation as `engine_core.hpp`: a declared-but-unused interface that reads as
configuration and configures nothing. If a C++-side config is wanted later it
should be added when something consumes it.

## B.2 Hardcoded values

| Finding | Location | Verdict |
| :--- | :--- | :--- |
| `file:///c:/Users/Anant/memforge/...` absolute links | `GETTING_STARTED_GUIDE.md:220` | **Fix.** Resolves for nobody but me — the same defect already fixed in `TEAM_EXECUTION_PLAN.md` |
| `DEFAULT_MODEL` not env-overridable | `benchmarks/run_prefix_sharing.py:47` | **Fix.** `run_live_engine.py:47` reads `MEMFORGE_TARGET_MODEL`; mine ignores it, so the two benchmarks can silently measure different models |
| `TARGET_MODEL` / `DRAFT_MODEL` hardcoded | `scripts/check_env.py:44-45` | **Fix.** The doctor should check for the weights you will actually load, which means honouring the same env vars the server reads |
| TinyLlama-1.1B defaults | `benchmarks/run_baseline.py`, `run_speculative.py`, `naive_runner.py` | **Fix the default, keep the flag.** These Phase-1 benchmarks default to a model the project does not cache, so running one triggers a surprise ~2 GB download. Default to the project's configured model; `--model` still overrides |
| Ports `8000` / `localhost:8000` | `run_stress_test.py`, `memforge_cli.py`, `traffic_simulator.py`, `vite.config.ts` | **Keep.** All are argparse defaults or the Vite dev proxy — overridable and conventional |
| Magic `16` (block size) | several call sites | **Keep.** Always alongside an explicit `block_size` parameter; `DEFAULT_BLOCK_SIZE` exists in `types.hpp` and is used where it matters |

## B.3 Stubs and unused imports

**Stubs:** no `TODO`, `FIXME`, `XXX`, `HACK` or `NotImplementedError` anywhere in
the source. The one `placeholder` hit is an HTML input attribute. Every bare
`pass` / `...` is either a documented exception swallow or a docstring example —
**no unimplemented functions remain**, `engine_core.hpp` having been the last one
and already deleted.

**15 unused imports**, all harmless but all noise:

```
memforge_ml/engine_bridge.py:6,7        Optional, time
memforge_ml/models/model_loader.py:6    os
memforge_ml/runtime/continuous_engine.py:25   SeqType
memforge_ml/speculative/rejection_sampler.py:8  Optional
memforge_ml/speculative/spec_engine.py:8        Optional
benchmarks/run_continuous_batching.py:9,16      time, MemForgeEngineBridge
benchmarks/run_speculative.py:17        NaiveBaselineRunner
tests/python/test_continuous_engine.py:10       torch
tests/python/test_paged_attention.py:2  pytest
tests/python/test_prefix_sharing.py:17  PagedKVPool
tests/python/test_rejection_sampler.py:2        pytest
setup.py:1                              os
```

`server/queues/request_queue.py:7` also has one, and goes away with the file.

## B.4 Things that are alive but worth knowing

Not defects; recorded so nobody "cleans up" something load-bearing.

- **`memforge_ml/models/paged_attention.py` is imported only by its own test.**
  Superseded in production by `runtime/paged_cache.py`. It is the standalone
  `PagedAttention` operator from Phase 1 and it is the natural reference
  implementation to validate Part A's tier 3 against — so it earns its place
  **as of this document**, and Part A should use it rather than leave it orphaned.
- **Two parallel speculative implementations.** `memforge_ml/speculative/*`
  (`spec_engine`, `draft_runner`, `rejection_sampler`) drives the simulation
  benchmark and pipeline stage 6; `runtime/speculative_runtime.py` is the live
  paged one. Both are reachable and tested. Worth consolidating eventually;
  deleting either today would break a pipeline stage.
- **`engine_bridge.py` is live** via `run_paged_kv.py` and pipeline stage 4, and
  `PyFallbackBlockAllocator` is its real fallback when `memforge_c` is absent.

## B.5 Known open bug, not hygiene

`EngineManager.speculative_stream` holds `_spec_lock` with `async with` across
its `yield`. When a client disconnects the generator closes, the lock releases —
**while the worker thread is still driving the single-pool
`PagedSpeculativeEngine`.** A second speculative request then starts a second
worker against the same KV pool. Confirmed with a minimal repro: worker still
running `True`, lock released `True`.

Not in Part A or B's scope, but it should not be forgotten, so it is written
down here. The fix needs a cancel flag plus a `threading.Lock` the worker itself
holds, because `await thread.join()` is illegal in a `finally` during
`GeneratorExit`.

## B.6 Audit summary

| Category | Count | Action |
| :--- | --: | :--- |
| Dead modules | 2 | delete (99 lines) |
| Dead definitions | 8 | 6 delete, 2 given a caller |
| Unsafe-but-live API | 1 | `free_sequence` made safe |
| Misleading declarations | 2 | `EngineConfig` deleted, `SWAPPED` comment corrected |
| Hardcoded values | 4 | fix; 2 categories deliberately kept |
| Unused imports | 16 | remove |
| Stubs / unimplemented | **0** | nothing to do |

---

# Part C — Implementation order

Part B first: it is small, mechanical and verifiable, and Part A should not be
built on top of code that is about to be deleted.

| Step | Work | Gate |
| --: | :--- | :--- |
| 1 | B.1 dead code, B.2 hardcoded values, B.3 imports | 157 tests + 7/7 pipeline |
| 2 | A.4 item 1 — `run_context_sweep.py` | Baseline recorded at all four context lengths |
| 3 | A.4 item 2 — `kernels/paged_gather.py` (tier 1a) | Byte-identical; sweep shows the delta |
| 4 | A.4 item 3 — `kernels/batched_decode.py` (tier 1b) | Byte-identical; the 640 MB/step copy gone |
| 5 | A.4 item 4 — tier 2 view cache, opt-in | Byte-identical, including under CoW, preemption and prefix hits |
| 6 | A.4 item 5 — tier 3 reference kernel | Equivalence proven against `paged_attention.py` |
| 7 | Re-run the sweep; record results, including any that miss the target | Docs updated with measured numbers |

**The standard, as everywhere else in this project:** a measured negative result
is a contribution, an unmeasured positive claim is not. §A.2.2 is already an
example — the plan's own suggested optimisation measured *slower* than the code
it was meant to replace, and that is written down rather than quietly skipped.
