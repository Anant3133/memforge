# MemForge on the A100 server

This replaces the hardware sections of `PHASE_E_RUNBOOK.md`, which describe a
local **RTX 4050 laptop on Windows**. This guide is for the **server**:

```
GPU     NVIDIA A100-SXM4-40GB (compute 8.0), shared with other tenants
CPU     256 logical cores, 1 TB RAM
OS      Linux 5.15, Python 3.12
driver  CUDA 12.6-class (560.x in-container)
torch   2.8.0+cu126   (installed inside the project venv, see below)
```

Everything is self-contained in `/workspace/memforge`: the virtualenv
(`.venv/`), the model cache (`.hf_cache/`), the built extension, the results.
Nothing outside the project folder is modified.

---

## 1. Setup from scratch

```bash
cd /workspace/memforge
python -m venv --system-site-packages .venv && . .venv/bin/activate

# The system torch is an NVIDIA "2.6.0a0" build that current transformers refuses
# (version check) and whose torchvision does not match. Install a stock torch
# *inside the venv*; it shadows the system one without touching it.
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
pip install -U transformers accelerate huggingface_hub fastapi uvicorn websockets httpx \
            pybind11 pytest pytest-timeout pytest-asyncio psutil matplotlib rich tqdm

python setup.py build_ext --inplace --force      # builds memforge_c (C++17)
export HF_HOME=$PWD/.hf_cache                    # models live inside the project
python scripts/check_env.py
CUDA_VISIBLE_DEVICES="" python -m pytest tests/python -q --timeout=300   # 398 + new tests
```

### Version constraints that cost time

| constraint | why |
|---|---|
| `transformers` **5.x** | `PagedCache` is written against the 5.x cache API. 4.57 fails with `Can't instantiate abstract class PagedCacheLayer ... get_max_cache_shape`. |
| `transformers` 4.5x/5.x **and** the system torch | fails: `cannot import name 'TransformGetItemToIndex' from torch._dynamo...`. Fixed by the venv torch. |
| torchvision must match torch | otherwise `operator torchvision::nms does not exist` and *every* model class import fails. |
| `pytest-rerunfailures` (system plugin) | can wedge a test run on a stuck server fixture; use `-p no:rerunfailures` if a run hangs. |

---

## 2. Models on this server

Cached under `.hf_cache/`. No Hugging Face token is configured and none is
needed; everything here is ungated.

| model | params | head_dim | KV heads | vocab | role |
|---|---:|---:|---:|---:|---|
| SmolLM2-135M-Instruct | 135M | 64 | 3 | 49152 | draft / smallest |
| SmolLM2-360M-Instruct | 360M | 64 | 5 | 49152 | original reference target |
| Qwen2.5-0.5B-Instruct | 0.5B | 64 | 2 | 151936 | draft |
| Qwen2.5-1.5B-Instruct | 1.5B | 128 | 2 | 151936 | target |
| Qwen2.5-3B-Instruct | 3B | 128 | 2 | 151936 | target |
| Qwen2.5-7B-Instruct | 7B | 128 | 4 | **152064** | target |
| Qwen3-0.6B / 1.7B / 4B | | 128 | 8 | 151936 | drafts |
| Qwen3-8B | 8B | 128 | 8 | 151936 | target |
| Qwen3-14B | 14B | 128 | 8 | 151936 | run via int8 weights on GPU (fp16 needs ~29.5 GB; ~25 GB was free) and fp32 on CPU |

fp16 weights: 7B ≈ 15 GB, 8B ≈ 16 GB. Both fit on the 40 GB card together with the
KV pool as long as the other tenants stay under ~20 GB.

---

## 3. Running the tests

```bash
scripts/run_model_ladder.sh <label> <device> <dtype> <target> [<draft>]
#   archives JSON + logs under benchmarks/results/runs/<label>/
KV=1024 CTXS="128 512 1024 2048" REPEATS=9 \
  scripts/run_model_ladder.sh gpu-r1-qwen7b cuda fp16 Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-0.5B-Instruct

scripts/run_gpu_all.sh                 # every model, 3 rounds, GPU (14B excluded)
scripts/run_full_ladder.sh             # CPU ladder (fp32, 32 threads)
scripts/run_improvement_experiments.sh # draft search, lookahead sweep, fp16/fp32 exactness, stress
python scripts/summarize_runs.py       # -> benchmarks/results/runs/AGGREGATE.md
```

Each benchmark takes `--device`, `--dtype`, `--out`; each model run writes
`prefix_sharing`, `context_sweep`, `chunked_prefill`, `speculative` JSON.

### Rules for trusting a GPU number on this machine

1. **The GPU is shared.** Check `nvidia-smi` before and after. Other tenants'
   memory and compute change run to run. Every harness checks for VRAM spill; a
   run that hits another tenant's memory fails with CUDA OOM rather than
   spilling (seen once: SmolLM2-360M speculative, 24 MB free).
2. **Contended timings are labelled.** During the first runs and for the
   rounds in `AGGREGATE.md` the card showed 100 % utilisation from a process
   that was not ours (see §5). Ratios between two arms measured back to back
   are still meaningful; absolute tokens/s are not.
3. **Use rounds, not single runs.** `summarize_runs.py` reports median and
   min-max across independent rounds. A single run that "passes" proves nothing.

---

## 4. Results (summary; full tables in `benchmarks/results/runs/AGGREGATE.md`)

Three independent GPU rounds per model (fp16), CPU ladder (fp32, 1 run each),
plus stress, fused-prefill and draft experiments. Details, evidence and the
failures are in **[KNOWN_FAILURES_AND_FIXES.md](KNOWN_FAILURES_AND_FIXES.md)**.

| area | A100 result |
|---|---|
| Correctness (paged = contiguous output) | **identical** on all 7 models, every run |
| Prefix sharing (N=4) | **0.333×** blocks, **0.27×** prefill work, 0 leaks, all models |
| Soak, 110 req @ concurrency 8 | **pass** on 0.5B, 1.5B, 3B, 7B, Qwen3-8B; 0 leaked blocks; RSS +4–25 MB |
| Paged vs contiguous speed | eager: 0.85× to 0.94×. **CUDA-graphed paged: 0.92-1.67× vs stock** (target met except 2048 tokens on 7B/8B, 0.93/0.92) |
| Chunked prefill, GPU | worst stall 3.0× → 2.2× (floor ~2.15×). Target 2.0× **not met** |
| Chunked prefill, CPU (4 runs) | budget 4 median 1.81-2.10×; under 2.0 on the median for 4 of 5 models, not in every run |
| Prefill/decode fusion, GPU | **worse** than unfused. Leave off |
| Speculative decoding, eager draft | 0.3-0.7× everywhere. **Net loss** |
| Speculative decoding, batched CUDA-graph draft | flat 0.82-0.91× across batch 1-8 (eager 0.70→0.35); no win on a contended card; 1.03× at batch 1 seen on a quiet card |
| fp16 'not lossless' | all 18 of 60 mismatches are exact logit ties (explained automatically) |
| Tests | CPU: 406 pass, 20 GPU-only skipped; GPU tests (graphs, batched draft, paged graph) pass on the A100 |

Full tables with n and ranges: [RESULTS_REPORT.md](RESULTS_REPORT.md).

Models run: SmolLM2-135M/360M, Qwen2.5-0.5B/1.5B/3B/7B, Qwen3-8B (draft pairs:
Qwen2.5-0.5B, Qwen3-0.6B/1.7B/4B) and **Qwen3-14B** (GPU int8: paged/contiguous 0.99, prefix sharing 0.333, identical output; CPU fp32).

One-command setup: `scripts/setup_server.sh [--dry-run]`. Regression gate: `python scripts/check_targets.py`
(`--update` accepts the current numbers as the new baseline). Open work: [TASKS.md](TASKS.md).

### How to reproduce a number

```bash
scripts/run_model_ladder.sh my-run cuda fp16 Qwen/Qwen3-8B Qwen/Qwen3-0.6B
python benchmarks/run_speculative_batching.py --model Qwen/Qwen3-8B --draft-model Qwen/Qwen3-0.6B \
       --device cuda --dtype fp16 --batch-sizes 1 2 4 --lookahead 3 --draft-cuda-graph
python scripts/summarize_runs.py
```

### Code changes made in this campaign

* `memforge_ml/runtime/batched_speculative.py`: vocab-padding fix; `GraphedDraftPool` /
  `GraphedDraftState` (CUDA-graph draft).
* `memforge_ml/runtime/continuous_engine.py`: `draft_cuda_graph` option, slot release on finish.
* `benchmarks/run_speculative_batching.py`: `--draft-cuda-graph`.
* New tests: `test_vocab_padding.py`, `test_graphed_draft.py`.
* New scripts: `run_model_ladder.sh`, `run_gpu_all.sh`, `run_full_ladder.sh`,
  `run_improvement_experiments.sh`, `run_fused_and_repeats.sh`,
  `run_graphed_draft_bench.sh`, `summarize_runs.py`; probes
  `benchmarks/probe_decode_latency.py`, `benchmarks/probe_cudagraph_draft.py`.

---

## 5. Known environment facts

* The card was shared with other jobs throughout (7.6 GB baseline held by two
  unrelated processes; a 30 GB job came and went).
* A GPU **crypto miner** (`peakminer`, supervised by
  `/tmp/.codex-gpu-until-stop-gpu2-*`) ran in this container for the whole
  campaign at ~370 W. Whoever owns the machine should confirm it is
  intended. It was **not** started or stopped by this project, and it inflates
  every GPU timing here.
* `nvidia-smi` lists other containers' processes as `[Not Found]`; their PIDs
  cannot be inspected from here.
