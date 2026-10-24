# MemForge Benchmark Report

Generated 2026-10-25 00:58:19 by `scripts/export_report.py`.
### Live engine end-to-end

Target `HuggingFaceTB/SmolLM2-360M-Instruct` · draft `HuggingFaceTB/SmolLM2-135M-Instruct` · cpu · torch.float32 · torch 2.13.0+cpu

**Continuous batching**

| Concurrency | Sequential s | Batched s | Speedup | Aggregate tok/s | Tokens/forward pass | Identical output |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.82 | 2.22 | 0.82x | 14.4 | 1.00 | yes |
| 2 | 3.85 | 2.65 | 1.46x | 24.2 | 1.89 | yes |
| 4 | 8.33 | 3.88 | 2.15x | 33.0 | 3.51 | yes |
| 8 | 15.09 | 4.55 | 3.31x | 56.2 | 6.29 | yes |

**Paging overhead** (median of 5 interleaved runs)

| Path | tok/s |
| ---: | ---: |
| Stock contiguous KV | 6.03 |
| MemForge paged KV | 5.89 |
| **Relative** | **0.98x** |

**Memory efficiency** — concurrent sequences that fit in 256.0 MB of KV

| Allocation strategy | Concurrent sequences |
| ---: | ---: |
| Contiguous reservation for max_model_len=1024 | 3 |
| Block-granular paging, 128 tokens used | **25** |
| **Capacity gain** | **8.33x** |

**Speculative decoding** (lookahead K=4)

| Metric | Value |
| ---: | ---: |
| Mean draft acceptance rate | 34.1% |
| Mean tokens per target forward pass | 2.26 |
| Output lossless vs greedy target | yes |
| **Mean wall-clock speedup** | **0.58x** |

> Speculative decoding is **0.58x, i.e. slower** on this CPU/model pair. The mechanism is verified lossless and confirms 2.26 tokens per target pass, but the 360M/135M cost ratio is too small for that to pay off in wall-clock terms on CPU. See PROGRESS_AND_NEXT_PHASES.md §3.5.

---

### Prefix sharing (Phase B)

Model `HuggingFaceTB/SmolLM2-360M-Instruct` · 512-token shared prefix · 12 tokens generated · block size 16

| Concurrency | Peak blocks | Ratio | Peak KV MB | Prefill tokens | Median TTFT | Wall s | Byte-identical |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 34 -> 34 | 1.000 | 42.5 -> 42.5 | 519 -> 519 | 5135 -> 5273 | 7.677 -> 7.339 | yes |
| 2 | 68 -> 36 | 0.529 | 85.0 -> 45.0 | 1038 -> 526 | 9515 -> 6011 | 17.702 -> 11.499 | yes |
| 4 | 136 -> 40 | 0.294 | 170.0 -> 50.0 | 2076 -> 540 | 14326 -> 6065 | 29.402 -> 15.905 | yes |
| 8 | 272 -> 48 | 0.176 | 340.0 -> 60.0 | 4152 -> 568 | 25546 -> 6787 | 54.072 -> 21.278 | yes |

Output was byte-identical to the non-sharing engine at every concurrency level.

Best memory reduction: **5.7x** at N=8.

---

### Sustained load & memory leak (Phase F)

100 requests · concurrency 8 · 12 tokens each · SSE streaming

| Measurement | Value |
| --- | --- |
| Requests | 110 ok / 110 sent |
| Status codes | `{'200': 110}` |
| Duration | 77.7 s |
| **Blocks leaked** | **0** |
| Blocks allocated, start -> end | 0 -> 0 |
| Peak blocks allocated | 22 |
| Peak blocks shared | 2 |
| Peak requests in flight | 8 |
| RSS initial | 1072.7 MB |
| RSS plateau (post-warm-up) | 1098.2 MB |
| RSS peak | 1195.2 MB |
| RSS final | 1113.5 MB |
| RSS growth vs plateau | 15.3 MB |
| TTFT p50 / p95 / p99 | 154.3 / 975.0 / 999.8 ms |
| Mean tokens/sec per request | 8.91 |
| Total tokens generated | 1320 |

**Verdicts**

- PASS  no leaked blocks at any sample
- PASS  all physical blocks returned to the pool
- PASS  RSS flat after warm-up (+15.3 MB)

Overall: **PASS**

> Block accounting is the exact measurement here: `blocks_leaked` is `total - free - allocated` and has no warm-up behaviour, so a nonzero value is a genuine leak. RSS is indicative only -- PyTorch and the allocator both cache, so it rises over the first requests and then plateaus, which is why growth is measured against the post-warm-up plateau rather than the first sample.
