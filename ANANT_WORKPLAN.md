# Anant — Track 3 Work Plan: Core Systems, Prefix Sharing & Server Hardening

**Owner:** Anant Mathur (Engineering Lead, Track 3)
**Branch:** `feat/anant-prefix-caching`
**Scope:** Phase B (prefix sharing via copy-on-write) + Phase F (reliability hardening)
**Written:** 2026-10-24 (sprint day 10 of 20)
**Reads with:** [TEAM_EXECUTION_PLAN.md](TEAM_EXECUTION_PLAN.md), [PROGRESS_AND_NEXT_PHASES.md](PROGRESS_AND_NEXT_PHASES.md), [CURRENT_STATUS.md](CURRENT_STATUS.md)

---

## 0. Accordance check against the team execution plan

Run before planning anything, on 2026-10-24. Two separate questions: *is the
baseline the plan assumes actually real?* and *has any sprint work landed?*

### 0.1 The baseline is real — verified, not assumed

| Claim in the plan / status docs | Verification run today | Result |
| :--- | :--- | :--- |
| 78 Python tests passing | `python -m pytest tests/python/ -q` | **78 passed** in 40.4 s |
| Native extension built and importable | `memforge_c.cp312-win_amd64.pyd` present, imported by the suite | **Present** |
| C++ core: allocator, scheduler, sequence | `engine/src/` — 4 sources, 5 headers | **Present** |
| `PhysicalBlock::ref_count` infrastructure exists and is unused | [block_allocator.cpp:33](engine/src/block_allocator.cpp#L33) sets `ref_count = 1`, nothing ever increments it | **Confirmed — exactly as documented** |
| Server drives one shared engine loop | [engine_manager.py](server/engine_manager.py) | **Present** |
| Frontend typechecks and builds | `frontend/tsconfig.json` + `dist/` artifacts committed | **Present** |

The 40 s suite runtime vs the "~8 s" in `PROGRESS_AND_NEXT_PHASES.md` §5 is a
cold-cache / machine-load difference, not a regression — the pass count is
identical. Not worth chasing.

### 0.2 Sprint deliverables — nothing has landed, on any track

The sprint started 2026-10-15. Today is day 10 of 20. **Milestone 1 (day 5,
Oct 19) was missed in full, and Milestone 2 is due tomorrow (Oct 25) with zero
of its four criteria met.**

| Track | Owner | Planned files | On disk today |
| :--- | :--- | :--- | :--- |
| 1 — Tooling | Ashutosh | `benchmarks/run_stress_test.py`, `scripts/memforge_cli.py`, `scripts/check_env.py`, `scripts/export_report.py` | **0 of 4** |
| 2 — ML kernels | Akshit | `memforge_ml/kernels/`, `memforge_ml/runtime/paged_gather.py`, `benchmarks/run_context_sweep.py` | **0 of 3** — no `kernels/` directory |
| 3 — Core & server | **Anant (me)** | `engine/{include/memforge,src}/prefix_cache.*`, `memforge_ml/runtime/prefix_sharing.py`, `server/api/disconnect_handler.py` | **0 of 4** |
| 4 — Frontend | Sanskar | `PrefixVisualizer.tsx`, `BenchmarkView.tsx`, `StatusBanner.tsx`, `src/views/` | **0 of 4** — no `views/` directory |
| 5 — QA & docs | Bhumika | `scripts/traffic_simulator.py`, `tests/python/test_edge_scenarios.py`, `docs/API_TESTING_GUIDE.md`, `docs/TEAM_ONBOARDING.md` | **0 of 4** — no `docs/` directory |

### 0.3 Process deviations from the plan's own rules

| Plan rule | Actual state | Severity |
| :--- | :--- | :--- |
| §1.3 "No direct commits to `main`" | `30772ec work assigned` went straight to `main` | Low — it is the plan document itself, a lead action |
| §6 "Create your dedicated feature branch" | `git branch -a` shows **only `main`** locally and on `origin`. Not one of the five `feat/*` branches exists | **High — nobody has started** |
| §6 "PRs require 1 review approval" | Zero PRs opened | High — follows from the above |

### 0.4 Defects in the plan document itself

These are mine to fix, since I own the doc:

1. **Broken cross-references.** The header links point at
   `file:///Users/ashutoshsharma/memforge/...` — a macOS path belonging to a
   different teammate. They resolve for nobody. Should be relative links.
2. **`memforge_ml/runtime/paged_cache.py` and `continuous_engine.py` are
   unowned, and both Akshit and I must edit them.** This directly violates
   §1.1 "no two developers will edit the same file". Akshit's in-place attention
   read (Phase A, limitation 2) rewrites the gather path inside `paged_cache.py`;
   my partial prefill needs `PagedCache.seed_position()` in the same file and a
   changed prefill branch in `continuous_engine.py`. **This is the single most
   likely merge conflict in the sprint.** Resolution in §6.1 below.
3. **`EngineConfig.num_gpu_blocks` default of 1024 is dead.** Nothing in the
   Python path reads it; the pool is sized from `kv_budget_mb`. Cosmetic, but it
   misleads anyone reading `types.hpp` first.
4. **Four build files hardcode the C++ source list**, so adding
   `prefix_cache.cpp` is a four-file change, not one: `setup.py`,
   `engine/CMakeLists.txt`, `scripts/cpp_test_runner.py`, `tests/cpp/CMakeLists.txt`.
   The plan's §1.5 "additive changes only on shared configs, coordinated through
   Anant" covers it — I am the coordinator, so these are mine.

### 0.5 Verdict

**Architecturally in accordance; schedule-wise we are not.** The plan's
technical premises all hold — the ref-counting hooks really are sitting unused,
the ownership split really does isolate cleanly (with the one exception in
§0.4.2), and the measured baseline is honest. But the sprint has produced no
code in 10 days. Milestone 2 cannot be met tomorrow. My own track is the
critical path for two other people (§6.2), so it goes first.

---

## 1. Who I am on this project, and what I own

I am the engineering lead and Track 3 owner. In practice that means three
different jobs, and it is worth being explicit about which hat is on:

**As Track 3 engineer** I own the C++ core, the paged-memory model, and the
FastAPI runtime. This is the part of MemForge where correctness is
unforgiving — a block freed twice or a ref-count missed does not throw, it
silently serves one request's KV cache to a different request and produces
plausible garbage. Everything in §3 and §4 is written with that in mind.

**As integration owner** I own the shared contracts in `TEAM_EXECUTION_PLAN.md`
§4, the shared build files, and the merge order. Two of my four deliverables
are dependencies for other people's definitions of done (§6.2), which is why
sequencing matters more than raw output here.

**As reviewer** I approve C++ engine, server-core, and documentation PRs.

### 1.1 My file ownership — exhaustive

Files I create:

| Path | Purpose |
| :--- | :--- |
| `engine/include/memforge/prefix_cache.hpp` | Chained block hashing + hash→physical registry |
| `engine/src/prefix_cache.cpp` | Its implementation |
| `memforge_ml/runtime/prefix_sharing.py` | Python-side prefix bookkeeping, CoW pool copies, savings telemetry |
| `server/api/disconnect_handler.py` | Reusable client-disconnect → scheduler-abort plumbing |
| `tests/cpp/test_prefix_cache.cpp` | C++ unit suite for hashing, sharing, eviction |
| `tests/python/test_prefix_sharing.py` | End-to-end sharing / CoW / byte-identity suite |
| `benchmarks/run_prefix_sharing.py` | The benchmark that proves the Phase B DoD number |

Files I modify:

| Path | Change |
| :--- | :--- |
| `engine/include/memforge/types.hpp` | Add `SequenceStepInfo::num_cached_tokens`; `PhysicalBlock::block_hash` |
| `engine/include/memforge/block_allocator.hpp` + `.cpp` | Ref-count increment path, zero-ref reporting, CoW `prepare_write` |
| `engine/include/memforge/scheduler.hpp` + `.cpp` | Prefix lookup at admission; forward zero-ref releases to the registry |
| `engine/src/bindings.cpp` | Expose `PrefixCache`, new fields, new allocator methods |
| `memforge_c.pyi` | Stubs for all of the above |
| `memforge_ml/runtime/paged_cache.py` | `seed_position()` for partial prefill (**shared with Akshit — see §6.1**) |
| `memforge_ml/runtime/continuous_engine.py` | Partial-prefill branch, sharing telemetry (**shared with Akshit**) |
| `server/engine_manager.py` | Backpressure semaphore, abort-on-disconnect, extended `live_metrics()` |
| `server/api/routes.py` | `Request` threading, 429 on saturation, `/metrics` endpoint |
| `setup.py`, `engine/CMakeLists.txt`, `tests/cpp/CMakeLists.txt`, `scripts/cpp_test_runner.py` | Register the new C++ source and test suite |
| `PROGRESS_AND_NEXT_PHASES.md`, `CURRENT_STATUS.md`, `TEAM_EXECUTION_PLAN.md` | Record results and fix §0.4 defects |

### 1.2 What I explicitly do NOT touch

`memforge_ml/kernels/*` and `paged_gather.py` (Akshit) · `frontend/**` (Sanskar) ·
`benchmarks/run_stress_test.py`, `scripts/memforge_cli.py`, `check_env.py`,
`export_report.py` (Ashutosh) · `scripts/traffic_simulator.py`,
`tests/python/test_edge_scenarios.py`, `docs/**` (Bhumika).

If I need a change in any of those, it goes through the owner as a request, not
a commit.

---

## 2. The problem I am solving, stated precisely

Today every request stores its own copy of its prompt's KV cache. Eight clients
hitting the same 512-token system prompt store that prompt eight times. From
`PROGRESS_AND_NEXT_PHASES.md` §7.5: *"`PhysicalBlock` carries a `ref_count` and
the allocator maintains it, but nothing yet shares blocks between sequences."*

The arithmetic for this model (81,920 bytes of KV per token, 16 tokens/block ⇒
1.25 MB per block, 204 blocks in a 256 MB pool):

| | Blocks for 8 × 512-token shared prompt | KV bytes |
| :--- | ---: | ---: |
| Today | 8 × 32 = **256** (exceeds the 204-block pool — requests get preempted) | 320 MB |
| With prefix sharing | 32 shared + 8 private tails = **40** | 50 MB |

That is a **6.4× reduction**, and it turns a workload that currently thrashes
the allocator into one that fits with room to spare. It is the largest remaining
memory win in the project and it needs no new hardware, no kernel work, and no
GPU.

There is a second, subtler win: the shared prefix's KV is *already computed*, so
a cache hit also skips the prefill forward pass for those tokens. That is a
latency win on TTFT, and it is why §3.4 (partial prefill) is not optional
polish — without it, sharing saves memory but recomputes the same numbers into
blocks that already hold them.

---

## 3. Phase B — prefix sharing via copy-on-write

### 3.1 Design decision: only full blocks are shareable

A block's Key/Value tensors depend on **every token before it**, because
attention is causal and cumulative. So two sequences may share physical block *i*
only if their token streams are identical for all of blocks *0..i*. Two
consequences:

1. **The hash must be chained, not per-block.**
   `h(i) = mix(h(i-1), tokens_of_block_i)`. A plain hash of block *i*'s own 16
   tokens would match two sequences that happen to share a middle fragment but
   differ earlier — and their K/V would be completely different. Getting this
   wrong produces fluent, confident, wrong output, which is the worst possible
   failure mode because no test that only checks for crashes will catch it.
2. **Sharing stops at the first miss.** If block 3 differs, blocks 4+ cannot be
   shared even if their tokens coincide. The lookup loop breaks, it does not
   continue scanning.

Only blocks holding a full `block_size` tokens are registered. A partially
filled block is still being written to, so its contents are not yet final and
its hash is not yet meaningful. The trailing partial block of every prompt is
always private. **This is what keeps the common path CoW-free** — decode only
ever writes the trailing block, which is private by construction.

### 3.2 Collision handling: verify, do not trust

A 64-bit hash over 16 token ids has a negligible accidental collision rate, but
"negligible" and "produces silently wrong text" is a bad combination. So the
registry stores the block's token ids alongside the hash and **memcmps on
lookup**. 16 `int32` is 64 bytes; the comparison is free relative to a forward
pass, and it reduces an entire class of nondeterministic wrongness to zero. This
is a deliberate trade of a few bytes per block for certainty.

Hash function: a chained splitmix64/FNV-1a hybrid implemented inline in the
header. No xxHash or Murmur dependency — the plan suggests them, but pulling a
third-party library into a project whose build already has four hardcoded source
lists (§0.4.4) and a broken g++ linker is a poor trade for a hash whose output
we verify anyway.

### 3.3 C++ components

**`engine/include/memforge/prefix_cache.hpp` / `.cpp`** — new

```cpp
namespace memforge {

uint64_t hash_block(uint64_t parent_hash, const TokenId* tokens, size_t n);

struct CachedBlock {
    BlockId block_id{INVALID_BLOCK_ID};
    uint64_t block_hash{0};
    std::vector<TokenId> tokens;   // for collision verification (§3.2)
    uint64_t last_hit_timestamp{0};
};

class PrefixCache {
public:
    explicit PrefixCache(std::shared_ptr<BlockAllocator> allocator, size_t block_size);

    // Walk `tokens` block-by-block, reusing physical blocks whose chained hash
    // AND token ids match. Stops at the first miss. Bumps ref_count on each hit.
    // Returns the reused block ids; caller allocates the remainder.
    std::vector<BlockId> acquire_shared_prefix(const std::vector<TokenId>& tokens);

    // Register a now-full block so later sequences can hit it.
    void register_block(BlockId id, uint64_t parent_hash,
                        const TokenId* tokens, size_t n);

    // Called with blocks whose ref_count reached 0. Removes their registry
    // entries so a recycled physical block is never served for stale content.
    void evict(const std::vector<BlockId>& released);

    size_t num_cached_blocks() const;
    size_t num_shared_blocks() const;   // entries with ref_count > 1
    double hit_rate() const;

private:
    std::shared_ptr<BlockAllocator> allocator_;
    size_t block_size_;
    std::unordered_map<uint64_t, CachedBlock> by_hash_;
    std::unordered_map<BlockId, uint64_t> by_block_;   // reverse, for evict()
    uint64_t hits_{0}, misses_{0};
    mutable std::mutex mutex_;
};

} // namespace memforge
```

**The reverse map `by_block_` is the load-bearing safety mechanism.** Without
it, a sequence finishes, its last reference drops, the block returns to the free
list, gets reallocated to an unrelated request — and the registry still points
at it. The next hash hit hands out a block full of someone else's conversation.
`evict()` closes that hole, and §5.2 has the test that proves it.

**`block_allocator.hpp` / `.cpp`** — modified

Four additions. The existing public signatures do not change, because 14 bridge
tests and the `.pyi` depend on them:

```cpp
// Increment an already-allocated block's ref_count (prefix hit).
bool add_ref(BlockId block_id);

// Free, returning only those ids whose ref_count actually reached zero.
// The caller forwards these to PrefixCache::evict() OUTSIDE the allocator lock.
std::vector<BlockId> release(const std::vector<BlockId>& block_ids);

// Copy-on-write: if the block at `logical_index` has ref_count > 1, allocate a
// private replacement, rewrite the sequence's block table, drop the shared ref.
// Returns {old_physical, new_physical} so the Python layer copies pool slots,
// or std::nullopt when the block was already private (the common case).
std::optional<std::pair<BlockId, BlockId>> prepare_write(Sequence& seq, size_t logical_index);

// Share-aware accounting for telemetry.
size_t get_num_shared_blocks() const;
```

**Lock ordering is `PrefixCache::mutex_` → `BlockAllocator::mutex_`, never the
reverse.** `BlockAllocator` must never call back into `PrefixCache` while
holding its own lock — that is why `release()` *returns* zero-ref ids instead of
evicting them itself. This is the kind of thing that works fine in every test and
deadlocks in the demo, so it is a hard rule, written in a comment at both call
sites.

**`scheduler.cpp`** — modified, at the admission branch
([scheduler.cpp:60-83](engine/src/scheduler.cpp#L60-L83))

Current admission allocates `get_num_required_blocks()` fresh blocks. New
admission:

```
tokens        = seq->get_all_tokens()
shared        = prefix_cache_->acquire_shared_prefix(tokens)   // may be empty
needed        = required_blocks - shared.size()
if (!allocator_->can_allocate(needed)) {
    prefix_cache_->release_acquired(shared);   // do not strand the refs we took
    break;                                     // retry next step
}
for (id : shared)                 seq->add_block(id);
for (id : allocator_->allocate_blocks(needed)) seq->add_block(id);
seq->set_num_cached_tokens(shared.size() * block_size);
```

The `release_acquired` on the failure path is easy to forget and leaks
ref-counts permanently when missed — blocks that can never be freed because
their count never returns to zero. Covered by a test in §5.2.

The preemption branch ([scheduler.cpp:36-46](engine/src/scheduler.cpp#L36-L46))
already calls `free_sequence`. It changes to use `release()` and forward the
zero-ref ids to `evict()`. A preempted sequence's shared blocks correctly
survive if another sequence still references them — which is a genuine
improvement to preemption, not just a sharing side effect.

**`types.hpp`** — modified

```cpp
struct PhysicalBlock {
    // ...existing...
    uint64_t block_hash{0};      // 0 = not registered in the prefix cache
};

struct SequenceStepInfo {
    // ...existing...
    size_t num_cached_tokens{0}; // prefix tokens whose KV is already in the pool
};
```

`num_cached_tokens` is the whole interface between the C++ sharing decision and
the Python prefill. It must also go into `bindings.cpp` and `memforge_c.pyi`,
and Sanskar's telemetry contract reads it indirectly.

### 3.4 Partial prefill — the part that is easy to underestimate

With `num_cached_tokens = C`, `_run_prefill` must stop recomputing the first `C`
tokens. Today it does this
([continuous_engine.py:300-312](memforge_ml/runtime/continuous_engine.py#L300-L312)):

```python
ids = torch.tensor([rt.tokens], ...)                 # the WHOLE prompt
out = self.model(input_ids=ids, past_key_values=rt.cache, use_cache=True)
```

With a cached prefix it becomes:

```python
cached = info.num_cached_tokens
rt.cache.seed_position(cached)                       # NEW: layers start at C
suffix = rt.tokens[cached:]
ids    = torch.tensor([suffix], ...)
pos    = torch.arange(cached, len(rt.tokens), ...)   # RoPE must see true positions
out = self.model(
    input_ids=ids,
    past_key_values=rt.cache,
    position_ids=pos.unsqueeze(0),
    cache_position=pos,
    use_cache=True,
)
```

Four traps here, each of which produces *plausible* output rather than an error:

1. **`position_ids` must be absolute** — but this turned out *not* to be an
   independent hazard. **Measured, 2026-10-24:** removing both `position_ids`
   and `cache_position` from the partial-prefill call changes nothing, because
   `transformers` derives the cache position from the cache's reported length
   and `PagedCache.get_seq_length()` returns the seeded `position` correctly.
   Trap 2 below subsumes this one. The explicit positions are kept as
   belt-and-braces — the engine knows the absolute offset for certain at that
   point and stating it does not depend on how a future `transformers` release
   infers it — but they are not what makes partial prefill correct. The
   original claim here overstated the risk; recorded rather than quietly
   dropped.
2. **`PagedCacheLayer.position` must start at `C`.** It is `0` in `__init__`
   ([paged_cache.py:36](memforge_ml/runtime/paged_cache.py#L36)) and the write
   range is derived from it, so without `seed_position` the suffix overwrites
   the shared prefix — corrupting it for every other sequence pointing at those
   blocks. The worst bug available in this design, **and the one that actually
   matters**: deleting the `seed_position` call fails the byte-identity tests
   immediately (verified by mutation, 2026-10-24). This single call is what
   makes partial prefill correct.
3. **The attention mask sizing already works.** `get_mask_sizes` returns
   `self.position + query_length`
   ([paged_cache.py:62](memforge_ml/runtime/paged_cache.py#L62)), so once
   `position == C` the causal mask spans cached + new correctly. No change
   needed — but verify it rather than assume it.
4. **`cached == len(rt.tokens)` is possible.** A prompt that is an exact
   multiple of 16 and fully cached leaves an empty suffix. A zero-length forward
   pass is undefined behaviour in `transformers`. Guard: always leave the last
   block private, so the suffix is never empty. This is already implied by §3.1
   (full blocks only) — but the guard is explicit and asserted.

**A preempted sequence re-enters through this same path** with
`rt.tokens = prompt + generated_so_far`. Prefix sharing and preempt-by-recompute
therefore interact, and the equivalence test in
`test_continuous_engine.py` ("preemption + recompute equivalence") must still
pass unchanged. It is the regression canary for this whole section.

### 3.5 Copy-on-write

With full-block-only sharing, decode writes only ever target the private
trailing block, so CoW should never fire in the common path. It is implemented
anyway, for three reasons: it makes partial-block sharing (§7, stretch) a safe
follow-up; it protects against a future scheduler change that shares a non-final
block; and an unexercised safety mechanism is worthless, so it gets a test that
forces the condition deliberately.

Flow, on every write to logical block *i*:

```
cow = allocator.prepare_write(seq, i)      # C++: usually std::nullopt
if cow is not None:
    old, new = cow
    pool.copy_block(old, new)              # Python: all layers, both K and V
    cache.set_block_table(seq.block_table) # rebind, invalidate the index memo
```

`PagedKVPool.copy_block(src, dst)` is a new method in `prefix_sharing.py` rather
than `kv_pool.py`, to keep the file off Akshit's path:

```python
key_pool[:, dst] = key_pool[:, src]
value_pool[:, dst] = value_pool[:, src]
```

Note `cache.set_block_table` must be called, not just the table mutated —
`PagedCache._index_memo` caches the logical→physical translation
([paged_cache.py:88-96](memforge_ml/runtime/paged_cache.py#L88-L96)) and a stale
memo after CoW writes to the *old* shared block. `set_block_table` already
clears the memo; relying on that is correct but needs a comment saying why,
because it looks like a redundant call.

### 3.6 Lifetime policy

**Stage 1 (this sprint):** a registry entry lives exactly as long as some
sequence references its block. Last reference drops ⇒ `evict()` ⇒ entry gone.
Simple, impossible to serve stale content, and sufficient for the Phase B DoD,
which is about *concurrent* requests.

**Stage 2 (stretch, §7):** retain zero-ref entries in a "cached but free" state
so a system prompt stays warm *between* requests, with LRU eviction when the
free list empties. That is where prefix caching pays off in real serving, and it
is also where the subtle bugs live — `allocate()` has to prefer never-cached
blocks and evict LRU cached ones only under pressure. Not worth risking the
sprint for; explicitly deferred and recorded as such.

### 3.7 Telemetry — additive only, so Sanskar never conflicts

`TEAM_EXECUTION_PLAN.md` §4 contract 2 asks for `shared_blocks`,
`memory_saved_mb`, and `block_owners: {block: [seq_ids]}`. Today
`block_occupancy()` returns `List[int]` — one owner per block
([continuous_engine.py:547-559](memforge_ml/runtime/continuous_engine.py#L547-L559)) —
which cannot express a shared block, and `MemoryVisualizer.tsx` consumes exactly
that shape.

**I keep `occupancy` as-is** (reporting the lowest-numbered owner) **and add new
keys beside it.** Sanskar's component keeps working untouched while he builds
`PrefixVisualizer.tsx` against the new keys on his own branch. Changing the
existing shape would break his branch mid-sprint for no benefit.

New keys on `/v1/engine/telemetry` and the `/ws/telemetry` feed:

```json
{
  "shared_blocks": 16,
  "cached_prefix_blocks": 32,
  "prefix_hit_rate": 0.78,
  "memory_saved_mb": 1.25,
  "block_owners": { "0": [1, 2, 3], "1": [1], "2": [3] }
}
```

`memory_saved_mb` = (sum of ref_counts − number of distinct blocks) ×
`bytes_per_block`. That is the honest figure: bytes we would have spent without
sharing, minus what we actually spent.

---

## 4. Phase F — server hardening

### 4.1 HTTP disconnect → scheduler abort

`PROGRESS_AND_NEXT_PHASES.md` §8 Phase F: *"the WebSocket path aborts; HTTP
disconnects are not yet detected."* Two distinct gaps:

**Streaming (SSE).** `EngineManager.stream()` has a `finally` that aborts
([engine_manager.py:253-257](server/engine_manager.py#L253-L257)), and Starlette
closes the generator on disconnect, so this *probably* works already. "Probably"
is not good enough for a reliability claim — it gets a test that asserts blocks
return to the free pool after a mid-stream client hangup.

**Non-streaming.** This one is definitely broken.
`create_chat_completion` awaits `ENGINE_MANAGER.generate(...)`
([routes.py:184](server/api/routes.py#L184)), which awaits a queue until the
sequence finishes. If the client hangs up, nothing notices: the sequence
generates all 96 tokens into a response no one will read, holding physical
blocks the whole time. Under Bhumika's 50-concurrent edge test that is a
self-inflicted memory-pressure event.

**`server/api/disconnect_handler.py`** — new, mine:

```python
async def run_cancellable(request: Request, coro, on_cancel):
    """
    Race `coro` against the client going away.

    Returns the coroutine's result, or raises ClientDisconnected after invoking
    `on_cancel` — which is how the engine learns to abort the sequence and free
    its blocks. Without this the sequence runs to completion unobserved.
    """

class AbortGuard:
    """Context manager: aborts `seq_id` unless the body completed normally."""
```

Implementation: `asyncio.wait({work, watcher}, FIRST_COMPLETED)` where the
watcher polls `request.is_disconnected()` on a short interval. Poll interval
250 ms — fast enough that blocks free within one or two engine steps, slow
enough to not add measurable load per in-flight request.

`routes.py` then threads `request: Request` into both completion handlers and
`_sse_stream`. Small diff, entirely inside files I own.

### 4.2 Backpressure

`ServerConfig.max_concurrency` defaults to 32
([config.py:55](server/config.py#L55)) **and is never read by anything.** Dead
config that looks like a working limit, which is worse than no config.

Add an `asyncio.Semaphore(max_concurrency)` in `EngineManager`, acquired in
`submit()`, released on stream close or abort. When saturated, return **429 with
a `Retry-After` header** rather than blocking — a blocked HTTP connection in a
saturated queue is how you turn load into timeouts.

This is a dependency for Bhumika: her "pool saturation (50 simultaneous requests
against capacity 25)" test needs a defined behaviour to assert, and right now
there is none.

### 4.3 `/metrics`

Phase F asks for structured logging and a `/metrics` endpoint. Plain-text
Prometheus exposition, no client library dependency:

```
memforge_blocks_total 204
memforge_blocks_allocated 48
memforge_blocks_shared 16
memforge_prefix_hit_rate 0.78
memforge_tokens_per_forward_pass 6.29
memforge_sequences_running 4
memforge_sequences_waiting 0
memforge_preemptions_total 2
```

Doubles as Ashutosh's scrape target for `run_stress_test.py`, so his harness
does not have to parse the JSON telemetry shape.

### 4.4 `EngineCore` — the decision

`engine/include/memforge/engine_core.hpp` declares a class with **no
implementation anywhere**. `bindings.cpp` includes the header
([bindings.cpp:8](engine/src/bindings.cpp#L8)) but binds nothing from it, which
is the only reason the link succeeds. `engine/CMakeLists.txt` lists it under
`ENGINE_HEADERS`. `PROGRESS_AND_NEXT_PHASES.md` §7.7 calls it "dead weight —
implement it or delete it."

**My recommendation: delete it.** Python already owns orchestration, and it owns
it *well* — `ContinuousBatchingEngine` has to be in Python because it drives
`model.forward()`. A C++ `EngineCore` could only wrap `Scheduler` +
`BlockAllocator`, which Python already composes directly in three lines
([continuous_engine.py:139-140](memforge_ml/runtime/continuous_engine.py#L139-L140)).
It would add a second orchestration path that nothing exercises, and
unexercised duplicate paths rot.

The counter-argument is optics: a C++ `EngineCore` reads well in a systems
project. I do not think it survives contact with "is this code ever run?" — and
the plan's own §7 standard is honesty about what is real. Deleting it, with the
rationale recorded in `PROGRESS_AND_NEXT_PHASES.md`, is the stronger statement.

**This is a judgement call, not a technical necessity — flagging it for
confirmation before I delete anything.** If the call goes the other way, the
minimum honest version is `EngineCore` as the owner of `PrefixCache` +
`BlockAllocator` + `Scheduler` lifetimes, bound to Python and actually used by
`ContinuousBatchingEngine.__init__` in place of the three separate handles. That
is real work, roughly a day, and it is Phase F scope.

---

## 5. Test plan

### 5.1 Correctness bar: byte-identical output

Every sharing test asserts **identical token ids**, not similar text, not close
logits. The whole class of bugs in §3.4 produces fluent wrong output, so
anything weaker than exact equality does not detect them. Seeded greedy
sampling, compare `token_ids` lists elementwise.

### 5.2 `tests/python/test_prefix_sharing.py` — new

| # | Test | Asserts |
| --: | :--- | :--- |
| 1 | Two identical prompts, sequential | Second allocates fewer blocks; outputs identical |
| 2 | Two identical prompts, concurrent | Shared blocks have `ref_count == 2`; both outputs identical to solo runs |
| 3 | **8 × 512-token shared prompt** | Allocated blocks ≈ 1/8 of the unshared run — **the Phase B DoD** |
| 4 | Shared prefix, divergent suffix | Sharing stops at the first differing block; neither output is affected |
| 5 | Common prefix shorter than one block | Zero sharing (correctly — nothing full to share) |
| 6 | Prompt an exact multiple of 16 | Last block stays private; suffix is never empty (§3.4 trap 4) |
| 7 | One sequence finishes, other continues | Survivor's blocks stay allocated; `ref_count` drops to 1 |
| 8 | Both finish | **Every block back in the free list; registry empty** |
| 9 | **Eviction correctness** | Fill the pool, free a shared block, reallocate it, re-hash the old prefix → **must miss, not hand back the recycled block** (the §3.3 hazard) |
| 10 | Forced CoW | Deliberately share a partial block; write; assert a private copy with identical prior contents |
| 11 | Sharing + preemption | Shrink the pool to force eviction with sharing on; output identical to the unpressured run |
| 12 | Admission failure mid-acquire | Prefix hit but insufficient blocks for the tail → **ref-counts fully released, nothing stranded** |
| 13 | Hash collision verification | Inject a colliding entry via the test hook; lookup rejects it on token mismatch |

### 5.3 `tests/cpp/test_prefix_cache.cpp` — new

Chained-hash determinism; different-prefix/same-block-tokens ⇒ different hash;
`acquire_shared_prefix` stops at first miss; `add_ref`/`release` arithmetic;
`evict` clears both maps; `prepare_write` returns `nullopt` for private blocks
and a pair for shared ones. Registered in `scripts/cpp_test_runner.py`
`CPP_TEST_SUITES` and `tests/cpp/CMakeLists.txt`.

### 5.4 `tests/python/` — hardening

Added to `test_api_stream.py` (shared file — additive only, coordinated):
mid-stream SSE disconnect frees blocks; non-streaming disconnect aborts the
sequence; 429 with `Retry-After` past `max_concurrency`; `/metrics` parses and
exposes the expected keys.

### 5.5 Non-negotiable regression gate

**All 78 existing tests must stay green.** Specifically
`test_continuous_engine.py` ("batched == sequential", "preemption + recompute
equivalence") and `test_scheduler_bridge.py` (14 allocator/scheduler tests) are
the ones most likely to break — they are the canaries for §3.4 and §3.3
respectively. `python scripts/run_pipeline.py` must pass all 7 stages before any
PR.

### 5.6 `benchmarks/run_prefix_sharing.py` — new

Produces the table that goes into `PROGRESS_AND_NEXT_PHASES.md`: 1/2/4/8
concurrent requests sharing a 512-token system prompt, with and without sharing,
reporting blocks allocated, KV MB, hit rate, TTFT, tok/s, and a byte-identity
check against the no-sharing run. Writes
`benchmarks/results/prefix_sharing.json` in the §4 contract-1 schema shape so
Ashutosh's `export_report.py` can consume it.

---

## 6. Interfaces I owe other people

### 6.1 The one real conflict in the plan, and how I am resolving it

`paged_cache.py` and `continuous_engine.py` are unowned and both Akshit and I
must change them (§0.4.2). My proposal, to confirm with him before either of us
starts:

- **I own both files** for this sprint, as Track 3 lead.
- Akshit's gather work lands in `memforge_ml/kernels/paged_gather.py` behind a
  **stable function boundary** that `paged_cache.py` calls:
  `gather_blocks(pool, layer_idx, physical, slots) -> (k, v)` and
  `scatter_blocks(pool, layer_idx, physical, slots, k, v) -> None` — the same
  shapes `PagedKVPool.gather`/`write` use today
  ([kv_pool.py:116-143](memforge_ml/runtime/kv_pool.py#L116-L143)).
- **I merge first** (prefix sharing changes *which* blocks get read), then he
  rebases and swaps the two call sites. Reverse order means he rebases a kernel
  onto a changed read-range contract, which is the harder direction.
- His in-place attention read needs a `paged_cache.py` change I cannot
  anticipate — so that specific edit comes to me as a PR against my branch, or I
  hand him the file after my merge. Agreed explicitly, not assumed.

### 6.2 Who is blocked on me

| Teammate | Needs from me | Their DoD item it unblocks |
| :--- | :--- | :--- |
| **Sanskar** | `shared_blocks`, `memory_saved_mb`, `block_owners` in telemetry (§3.7) | `PrefixVisualizer.tsx` — currently has nothing real to render |
| **Bhumika** | 429 backpressure behaviour (§4.2) | "Pool saturation, 50 requests vs capacity 25" — no defined behaviour to assert today |
| **Bhumika** | HTTP disconnect abort (§4.1) | "Rapid client disconnect right after prefill" |
| **Ashutosh** | `/metrics` (§4.3), `shared_blocks` accounting | `leaked_blocks` field in the stress-test schema |
| **Akshit** | `SequenceStepInfo.num_cached_tokens` (§3.3) | Partial prefill changes his gather read ranges |

**Three of five tracks have a definition of done that depends on my work.** That
is the argument for the sequencing in §8: ship the telemetry keys and the 429
early, even as stubs returning real-but-boring numbers, so nobody waits on the
full CoW implementation.

### 6.3 Contracts I publish

Per `TEAM_EXECUTION_PLAN.md` §4 contract 3, slightly corrected — the plan's
`compute_block_hash(token_ids) -> uint64` omits the parent hash, which §3.1
shows is the whole correctness argument:

```
# C++, via memforge_c
hash_block(parent_hash: int, tokens: List[int]) -> int
PrefixCache.acquire_shared_prefix(tokens: List[int]) -> List[int]
PrefixCache.num_shared_blocks() -> int
PrefixCache.hit_rate() -> float
BlockAllocator.add_ref(block_id: int) -> bool
BlockAllocator.release(block_ids: List[int]) -> List[int]   # those that hit zero
BlockAllocator.prepare_write(seq, logical_index) -> Optional[Tuple[int, int]]
SequenceStepInfo.num_cached_tokens: int
```

I will raise the signature correction in the plan doc rather than silently
shipping something different from what is written.

---

## 7. Risks, and what is explicitly out of scope

| Risk | Why it is dangerous | Mitigation |
| :--- | :--- | :--- |
| Un-chained hash shares the wrong block | Fluent, confident, wrong output. No crash | Chained hash (§3.1) + token verification (§3.2) + byte-identity tests (§5.1) |
| Stale registry entry after block recycling | Serves another conversation's KV | `by_block_` reverse map + `evict()` + test 9 |
| `position_ids` not absolute under partial prefill | Subtly wrong RoPE; grammatical output | §3.4 trap 1, test 1/2 byte-identity |
| Suffix overwrites a shared prefix | Corrupts every sequence sharing those blocks | `seed_position()` (§3.4 trap 2), test 10 |
| Lock inversion PrefixCache ↔ BlockAllocator | Deadlocks the server under concurrency, passes every unit test | `release()` returns ids; eviction outside the lock (§3.3) |
| Stranded ref-counts on admission failure | Blocks that can never be freed; slow death | `release_acquired` on the failure path, test 12 |
| Merge conflict with Akshit | Two branches rewriting the same read path | §6.1 ownership + merge order, agreed up front |
| Four hardcoded build source lists | `prefix_cache.cpp` silently not compiled; confusing link errors | §0.4.4 checklist, all four in one commit |
| g++ linker broken on this machine | Looks like a test failure | Known and documented; `cpp_test_runner.py` falls back to MSVC |

**Out of scope, deliberately:** zero-ref retained cache with LRU (§3.6 stage 2) ·
partial-block sharing · cross-process / disk prefix cache · GPU paths (Phase E) ·
chunked prefill (Phase C) · speculative-inside-batching (Phase D) · anything in
`frontend/`, `kernels/`, or `docs/`.

---

## 8. Sequencing

Day 10 of 20. Milestone 2 is tomorrow and will not be met (§0.5). Ordering is
driven by §6.2 — unblock other people before finishing my own deep work.

| Day | Date | Work | Gate |
| --: | :--- | :--- | :--- |
| 10 | Oct 24 | Branch; accordance check (this doc); `prefix_cache.hpp/cpp` + hashing; 4 build files | `test_prefix_cache.cpp` green |
| 11 | Oct 25 | `add_ref`/`release`/`evict`; scheduler admission; bindings + `.pyi` | 78 existing tests still green |
| 12 | Oct 26 | **Unblock day:** telemetry keys for Sanskar, 429 for Bhumika, `/metrics` for Ashutosh — shipped early even if sharing is incomplete | Three teammates unblocked |
| 13 | Oct 27 | `seed_position` + partial prefill + `prefix_sharing.py` | Tests 1–8 green |
| 14 | Oct 28 | CoW `prepare_write` + `copy_block`; eviction hardening | Tests 9–13 green |
| 15 | Oct 29 | `disconnect_handler.py`; HTTP + SSE abort paths | §5.4 green |
| 16 | Oct 30 | `run_prefix_sharing.py`; **prove the 1/8 DoD number**. Milestone 3 integration | Benchmark recorded |
| 17 | Oct 31 | `EngineCore` decision executed; plan-doc defects §0.4 fixed | Pipeline 7/7 |
| 18 | Nov 01 | Integration with Akshit's kernel and Sanskar's visualizer; review others' PRs | Cross-track green |
| 19–20 | Nov 02–03 | Progress/status docs updated with measured numbers; final demo | Milestone 4 |

Day 12 is the one that matters most to the team and is easiest to skip. It does
not get skipped.

### 8.1 Git workflow

```bash
git checkout main && git pull origin main
git checkout -b feat/anant-prefix-caching
# ... work ...
pytest tests/python/ && python scripts/cpp_test_runner.py
git push -u origin feat/anant-prefix-caching
# rebase, never merge:
git fetch origin && git rebase origin/main
```

Commits scoped one subsystem each (`feat(engine): ...`, `feat(server): ...`) so
a bad one can be reverted alone. Squash-merge on approval.

---

## 9. Definition of done

**Phase B**

- [ ] Chained block hashing with token verification, in C++, unit tested
- [ ] Identical logical blocks across concurrent sequences point at one physical block, `ref_count` bumped
- [ ] CoW allocates a private block and copies before any write to a shared block
- [ ] Partial prefill skips recomputing a cached prefix, with absolute `position_ids`
- [ ] **8 concurrent requests on a 512-token shared prompt allocate ≈1/8 the blocks**
- [ ] Output **byte-identical** to the non-sharing engine, every test
- [ ] `evict()` makes stale-registry reuse impossible — test 9 proves it
- [ ] `tests/python/test_prefix_sharing.py` 13/13; `tests/cpp/test_prefix_cache.cpp` green
- [ ] `benchmarks/results/prefix_sharing.json` recorded; table in the progress doc

**Phase F**

- [ ] HTTP client disconnect aborts the sequence and frees blocks immediately — streaming *and* non-streaming
- [ ] `max_concurrency` actually enforced; 429 + `Retry-After` on saturation
- [ ] `/metrics` endpoint live
- [ ] `EngineCore` resolved, with the decision written down
- [ ] No block leaks under sustained load: `free + allocated == total`, always

**Always**

- [ ] 78 existing tests green
- [ ] `python scripts/run_pipeline.py` 7/7
- [ ] Docs updated with measured numbers, including any that come out negative

---

## 10. Standard this work is held to

`PROGRESS_AND_NEXT_PHASES.md` reports that speculative decoding came out at
**0.58× — slower** — and explains why, instead of hiding it. That is the
standard. If prefix sharing turns out to cost more in hashing and bookkeeping
than it saves, or if the hit rate on realistic traffic is disappointing, that
goes in the document with the numbers and the reason. A measured negative result
is a contribution; an unmeasured positive claim is not.
