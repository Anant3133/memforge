# MemForge: Custom LLM Inference Engine from Scratch

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.3+-61dafb.svg)](https://react.dev/)

MemForge is a high-performance LLM inference engine built from scratch. It bridges
low-level C++ systems memory management with Python/PyTorch machine learning serving to
maximize throughput and eliminate memory waste during autoregressive text generation.

It runs real model weights end to end today. The headline measured results (CPU-only,
`SmolLM2-360M` target):

| | Measured |
| :--- | :--- |
| Continuous batching, 8 concurrent requests | **3.32x faster** than serving them one at a time (6.29 tokens per forward pass) |
| Concurrent sequences per byte of KV memory | **8.3x** more than contiguous reservation |
| Paged KV cache vs stock contiguous KV cache | **output bit-identical**, throughput break-even (0.98x) |
| Speculative decoding | **verified lossless** (greedy speculative == greedy target exactly); 2.26 tokens per target pass, but a net wall-clock *loss* at this model scale -- see the report |

Full results, limitations and the next phases:
[PROGRESS_AND_NEXT_PHASES.md](PROGRESS_AND_NEXT_PHASES.md).

---

## Key Features

1. **PagedAttention & BlockAllocator (C++)**: OS-style virtual memory paging for Key-Value caches (16-token fixed physical blocks) eliminating external fragmentation.
2. **Continuous Batching & Iteration Scheduler (C++)**: Step-by-step dynamic scheduling that ingests and evicts requests at every iteration without waiting for long sequences to finish.
3. **Speculative Decoding with Statistical Verification (PyTorch)**: Fast draft model generates candidate tokens, verified by the target model in a single forward pass with statistical rejection sampling.
4. **Real-time Telemetry Dashboard (React + WebSockets)**: Live tokens/sec, tokens per
   forward pass, and a physical block heatmap coloured by owning sequence -- which makes
   the scattered, non-contiguous layout of a paged sequence visible -- plus a colour-coded
   draft vs target token diff.

The paged cache is implemented as a `transformers.Cache`, so **any** causal Hugging Face
model's forward pass can be run on C++-allocated paged memory without patching its
attention code.

---

## Directory Layout

```
memforge/
├── engine/               # C++ Core Engine (BlockAllocator, Continuous Scheduler, pybind11)
├── memforge_ml/
│   ├── runtime/          # Live engine: kv_pool, paged_cache, continuous_engine, speculative_runtime
│   ├── models/           # Model loading + standalone PagedAttention operator
│   └── speculative/      # Rejection sampler & draft runner
├── server/               # FastAPI REST & WebSocket Server
├── frontend/             # React + Vite Telemetry & Streaming UI Dashboard
├── benchmarks/           # run_live_engine.py (end-to-end) + component microbenchmarks
├── tests/                # Automated C++ & Python test suites
├── scripts/              # run_pipeline.py (7-stage verification), cpp_test_runner.py
├── MEMFORGE_PLAN.md      # Master 9-week development roadmap & architecture specifications
├── PROGRESS_AND_NEXT_PHASES.md  # Measured results, limitations, next phases
├── CURRENT_STATUS.md     # Subsystem-by-subsystem status
├── CMakeLists.txt        # C++ build system
├── setup.py              # Python C++ extension build script
└── requirements.txt      # Python dependencies
```

---

## Quickstart

### 1. Environment setup

Create and **activate** a virtualenv before installing or downloading anything:

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# Git Bash / Linux / macOS
python -m venv .venv
source .venv/Scripts/activate   # or .venv/bin/activate
```

```bash
pip install -r requirements.txt

# Build the native C++ extension (required by the engine)
python setup.py build_ext --inplace
python -c "import memforge_c; print(memforge_c.BlockAllocator(128, 16).get_num_free_blocks())"
```

### 2. Download the models

Small by default so everything runs on CPU. Both share one tokenizer, which
speculative decoding requires.

```bash
# HF_HUB_DISABLE_SYMLINKS=1 avoids a Windows symlink-privilege error
HF_HUB_DISABLE_SYMLINKS=1 hf download HuggingFaceTB/SmolLM2-360M-Instruct
HF_HUB_DISABLE_SYMLINKS=1 hf download HuggingFaceTB/SmolLM2-135M-Instruct
```

### 3. Verify everything

```bash
# 7 stages: C++ tests, pybind11 bridge, 78 pytest, 3 microbenchmarks, live engine
python scripts/run_pipeline.py

# Or just the test suite
pytest
```

### 4. Benchmark on real weights

```bash
python benchmarks/run_live_engine.py --tokens 32 --concurrency 1 2 4 8
```

Writes `benchmarks/results/live_engine.json`. The older
`benchmarks/run_{paged_kv,continuous_batching,speculative,baseline}.py` scripts are
component-level microbenchmarks and simulations; `run_live_engine.py` is the
end-to-end measurement.

### 5. Start the server & dashboard

```bash
# Terminal 1 - FastAPI backend (loads models on startup)
uvicorn server.app:app --host 127.0.0.1 --port 8000

# Terminal 2 - React dashboard (proxies /v1 and /ws to port 8000)
cd frontend && npm install && npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

```bash
# Or talk to it directly
curl http://127.0.0.1:8000/v1/health
curl http://127.0.0.1:8000/v1/engine/telemetry
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is paging?"}],"max_new_tokens":48}'
```

### Configuration

Every setting is overridable by environment variable: `MEMFORGE_TARGET_MODEL`,
`MEMFORGE_DRAFT_MODEL`, `MEMFORGE_KV_BUDGET_MB`, `MEMFORGE_MAX_NUM_SEQS`,
`MEMFORGE_MAX_MODEL_LEN`, `MEMFORGE_LOOKAHEAD_K`, `MEMFORGE_LOAD_DRAFT`,
`MEMFORGE_BATCHED_DECODE`. See `server/config.py`.

---

## API

| Endpoint | Purpose |
| :--- | :--- |
| `POST /v1/chat/completions` | OpenAI-compatible; `stream: true` for SSE, `use_speculative: true` to route to the draft+target engine |
| `POST /v1/completions` | Raw-prompt completion (no chat template) |
| `GET /v1/models` | Available models |
| `GET /v1/health` | Readiness and load errors |
| `GET /v1/engine/status` | Full engine config, KV pool geometry, cumulative telemetry |
| `GET /v1/engine/telemetry` | Live counters |
| `GET /v1/engine/blocks` | Physical block ownership map for the heatmap |
| `WS /ws/stream` | Token stream with per-token provenance (target/draft/resample/bonus) |
| `WS /ws/telemetry` | Engine state pushed ~2.5x/second |

---

## Roadmap & Team Breakdown

- [PROGRESS_AND_NEXT_PHASES.md](PROGRESS_AND_NEXT_PHASES.md) — what is built and measured,
  known limitations, and the next phases (A: fused paged-attention kernel, B: prefix
  sharing, C: chunked prefill, D: speculation inside continuous batching, E: GPU, F: hardening).
- [CURRENT_STATUS.md](CURRENT_STATUS.md) — subsystem-by-subsystem status.
- [MEMFORGE_PLAN.md](MEMFORGE_PLAN.md) — the original 9-week sprint schedule, mathematical
  formulations and role assignments.
