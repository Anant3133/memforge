# MemForge: Custom LLM Inference Engine from Scratch

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.3+-61dafb.svg)](https://react.dev/)

MemForge is a high-performance LLM inference engine built from scratch. It bridges low-level C++ systems memory management with Python/PyTorch machine learning serving to maximize throughput and eliminate memory waste during autoregressive text generation.

---

## Key Features

1. **PagedAttention & BlockAllocator (C++)**: OS-style virtual memory paging for Key-Value caches (16-token fixed physical blocks) eliminating external fragmentation.
2. **Continuous Batching & Iteration Scheduler (C++)**: Step-by-step dynamic scheduling that ingests and evicts requests at every iteration without waiting for long sequences to finish.
3. **Speculative Decoding with Statistical Verification (PyTorch)**: Fast draft model generates candidate tokens, verified by the target model in a single forward pass with statistical rejection sampling.
4. **Real-time Telemetry Dashboard (React + WebSockets)**: Live visualization of Tokens/sec, GPU VRAM block heatmaps, and color-coded draft vs target token verification diffs.

---

## Directory Layout

```
memforge/
├── engine/               # C++ Core Engine (BlockAllocator, Continuous Scheduler, pybind11)
├── memforge_ml/          # PyTorch ML Layer (PagedAttention, Speculative Engine, Baseline)
├── server/               # FastAPI REST & WebSocket Server
├── frontend/             # React + Vite Telemetry & Streaming UI Dashboard
├── benchmarks/           # Benchmarking scripts (Baseline, Continuous Batching, Paged KV, Speculative)
├── tests/                # Automated C++ & Python test suites
├── MEMFORGE_PLAN.md      # Master 9-week development roadmap & architecture specifications
├── CMakeLists.txt        # C++ build system
├── setup.py              # Python C++ extension build script
└── requirements.txt      # Python dependencies
```

---

## Quickstart

### 1. Environment Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# (Optional) Build & install C++ native pybind11 extension
pip install -e .
```

### 2. Run Benchmarks

```bash
# 1. Paged KV-Cache allocation & memory fragmentation test
python benchmarks/run_paged_kv.py

# 2. Continuous batching vs static batching throughput simulation
python benchmarks/run_continuous_batching.py

# 3. Phase 1 Naive baseline test (unoptimized PyTorch loop)
python benchmarks/run_baseline.py --tokens 64

# 4. Speculative decoding speedup benchmark
python benchmarks/run_speculative.py --tokens 64
```

### 3. Run Automated Tests

```bash
# Run Python pytest suite (rejection sampling, paged attention, API endpoints)
pytest
```

### 4. Start the Inference Server & Dashboard

```bash
# Start FastAPI backend server (Port 8000)
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Start React Frontend Dashboard (Port 3000)
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser to interact with the MemForge telemetry dashboard.

---

## Roadmap & Team Breakdown

For detailed 9-week sprint schedules, mathematical formulations, and role assignments, see [MEMFORGE_PLAN.md](file:///c:/Users/Anant/memforge/MEMFORGE_PLAN.md).
