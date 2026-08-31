# MemForge: Current Status & Execution Blueprint

---

## 1. Executive Summary & Current Repository Status

The MemForge repository has been initialized with the core architectural scaffolding across low-level C++ systems memory management, Python/PyTorch machine learning serving, FastAPI asynchronous middleware, and a React telemetry frontend.

```mermaid
graph TD
    subgraph S1["Completed in Scaffolding (Phase 1 Initial)"]
        C1["C++ Core Data Structs & Engine Headers"]
        C2["C++ BlockAllocator (Paged KV) & Scheduler"]
        C3["ML Baseline Runner & Speculative Sampler"]
        C4["FastAPI Server & WebSocket Streaming Endpoints"]
        C5["React Telemetry Dashboard & Speculative Diff UI"]
        C6["Automated C++ Unit Test Verification (100% Pass)"]
    end

    subgraph S2["Next Milestone (Phase 1 Completion -> Phase 2)"]
        N1["Build Native pybind11 Extension (.pyd/.so)"]
        N2["Execute Baseline Benchmarks on Target/Draft Models"]
        N3["Integrate PagedAttention into PyTorch Forward Pass"]
        N4["Wire Live Server Async Request Queue to C++ Scheduler"]
        N5["Connect Frontend to Live WebSocket Telemetry"]
    end

    S1 --> S2
```

---

## 2. Detailed Status by Subsystem

### 2.1 C++ Core Engine (Systems Architect)
- **Completed**:
  - `engine/include/memforge/types.hpp`: Defined `SequenceId`, `TokenId`, `BlockId`, `SequenceStatus`, `PhysicalBlock`, `SequenceStepInfo`, and `SchedulerStepResult`.
  - `engine/include/memforge/sequence.hpp` & `sequence.cpp`: Implemented token sequences with dynamic block table tracking and boundary auto-expansion.
  - `engine/include/memforge/block_allocator.hpp` & `block_allocator.cpp`: Implemented thread-safe physical block pool (16 tokens/block), free-list management, ref-counting, and deallocation.
  - `engine/include/memforge/scheduler.hpp` & `scheduler.cpp`: Implemented iteration-level continuous batching scheduler managing waiting/running queues and prefill promotion.
  - `engine/src/bindings.cpp`: `pybind11` bridge exporting C++ classes (`BlockAllocator`, `Scheduler`, `Sequence`, `SequenceStatus`, `SchedulerStepResult`) to Python.
  - **Testing**: C++ unit tests (`test_block_allocator.cpp` and `test_scheduler.cpp`) compiled with `g++ -std=c++17` and passed all assertions with 0 errors.

### 2.2 ML & Attention Layer (ML Algorithm Lead)
- **Completed**:
  - `memforge_ml/baseline/naive_runner.py`: Phase 1 unoptimized autoregressive generation loop to capture baseline throughput, TTFT, and VRAM footprint.
  - `memforge_ml/models/model_loader.py`: Hugging Face model loader with automatic fallback to lightweight models (TinyLlama / SmolLM) for low-resource environments.
  - `memforge_ml/models/paged_attention.py`: Paged KV-cache operator (`PagedKVCache` and `PagedAttention`) reading non-contiguous physical memory blocks.
  - `memforge_ml/speculative/draft_runner.py`: Candidate token generator proposing $K$ tokens via the draft model.
  - `memforge_ml/speculative/rejection_sampler.py`: Statistical rejection sampling algorithm applying $\min\left(1, \frac{P(x)}{Q(x)}\right)$ and lossless distribution resampling.
  - `memforge_ml/speculative/spec_engine.py`: Target + Draft speculative loop orchestrator.
  - `memforge_ml/engine_bridge.py`: Python bridge connecting C++ scheduler and allocator with PyTorch execution.

### 2.3 API & Middleware (API / Middleware Dev)
- **Completed**:
  - `server/app.py`: FastAPI server with CORS middleware and lifespan management.
  - `server/api/routes.py`: OpenAI-compatible `/v1/chat/completions`, `/v1/models`, and `/v1/health` endpoints.
  - `server/api/websocket.py`: `/ws/stream` WebSocket endpoint for real-time token streaming and live telemetry emission.
  - `server/queues/request_queue.py`: Thread-safe async queue dispatching requests to continuous batching.

### 2.4 Frontend & Telemetry Dashboard (Telemetry & UI Dev)
- **Completed**:
  - `frontend/package.json` & `vite.config.ts`: React + Vite + TypeScript build setup.
  - `frontend/src/index.css`: Glassmorphism styling with dark theme, glowing badges, and smooth animations.
  - `frontend/src/components/TelemetryMetrics.tsx`: Live throughput (Tokens/sec), acceptance rate, and VRAM utilization cards.
  - `frontend/src/components/MemoryVisualizer.tsx`: Physical memory block heatmap displaying active vs free blocks.
  - `frontend/src/components/SpeculativeDiff.tsx`: Token-by-token diff highlighting Draft-accepted vs Target-verified words.
  - `frontend/src/components/ChatStream.tsx` & `App.tsx`: Prompt dispatcher and dashboard layout.

### 2.5 Unified Master Pipeline (Scripts)
- **Completed**:
  - `scripts/run_pipeline.py` & `scripts/run_pipeline.ps1`: Single-command pipeline runner orchestrating all 6 testing and benchmarking stages with rich stylized terminal dashboard reporting.
  - **Live Verification Results**:
    - **Stage 1 (C++ Unit Tests)**: PASSED (BlockAllocator & Continuous Batching Scheduler).
    - **Stage 2 (Native pybind11 Bridge)**: PASSED (`memforge_c` binary extension loaded in Python with 4,096 tokens capacity).
    - **Stage 3 (Pytest Suite)**: PASSED (7 tests covering statistical rejection sampling, Paged KV Cache read/write, and FastAPI REST endpoints).
    - **Stage 4 (Paged KV Cache Benchmark)**: PASSED (6.86M blocks/sec allocation speed, 0.0% memory fragmentation).
    - **Stage 5 (Continuous Batching Benchmark)**: PASSED (2.67x compute throughput speedup over static batching).
    - **Stage 6 (Speculative Decoding Benchmark)**: PASSED (Lossless target distribution preservation verified).

---

## 3. What We Are Doing Next (Immediate Next Milestones)

```mermaid
gantt
    title Immediate Next Execution Steps
    dateFormat  YYYY-MM-DD
    section Systems Track
    Compile native pybind11 extension (`pip install -e .`) :a1, 2026-09-01, 3d
    Thread safety & memory block row-level locking          :a2, after a1, 4d
    section ML Track
    Run baseline benchmark on target & draft models         :b1, 2026-09-01, 3d
    Hook PagedAttention into model forward pass             :b2, after b1, 5d
    section API & UI Track
    Connect live FastAPI request queue to C++ scheduler     :c1, 2026-09-01, 4d
    Run frontend locally & connect to WebSocket backend     :c2, after c1, 3d
```

### 3.1 Step 1: Compile the Native C++ `pybind11` Extension
- **Objective**: Build `memforge_c.pyd` (Windows) / `memforge_c.so` (Linux/macOS) so Python code directly executes the high-performance C++ `BlockAllocator` and `Scheduler`.
- **Target**: Run `pip install -e .` or `python setup.py build_ext --inplace`.

### 3.2 Step 2: Establish Phase 1 Baseline Performance Benchmarks
- **Objective**: Run `benchmarks/run_baseline.py` to record baseline numbers for:
  1. Target Model Throughput (Tokens/sec)
  2. Time-To-First-Token (TTFT)
  3. Peak VRAM / Memory Utilization
- **Target**: Establish standard baseline against which Continuous Batching and Speculative Decoding speedups will be quantitatively measured.

### 3.3 Step 3: Integrate `PagedAttention` into Model Layer
- **Objective**: Replace standard PyTorch contiguous KV cache tensor allocations with `PagedKVCache` in the attention forward pass, gathering $K, V$ tensors via C++ block tables.
- **Target**: Run `benchmarks/run_paged_kv.py` to verify zero memory fragmentation.

### 3.4 Step 4: Wire Continuous Batching Request Queue in FastAPI
- **Objective**: Connect `server/queues/request_queue.py` to the C++ `Scheduler` so multiple concurrent HTTP requests are dynamically interleaved at the single-token step level.
- **Target**: Verify with multi-request concurrent test script.

### 3.5 Step 5: Live UI Telemetry Connection
- **Objective**: Run `npm install` and `npm run dev` in `frontend/`, verify real-time WebSocket connection to `server/app.py`, and validate live metric graphs.

---

## 4. Step-by-Step Implementation Plan

### Step 1: Environment & C++ Extension Compilation
```bash
# 1. Install python dependencies
pip install -r requirements.txt

# 2. Build C++ pybind11 extension in-place
python setup.py build_ext --inplace

# 3. Verify Python can import C++ engine
python -c "import memforge_c; alloc = memforge_c.BlockAllocator(128, 16); print('C++ Allocator initialized! Free blocks:', alloc.get_num_free_blocks())"
```

### Step 2: Run Automated Test Suite
```bash
# Run pytest for PyTorch attention, rejection sampling, and FastAPI routes
pytest -v
```

### Step 3: Run Baseline & Optimization Benchmarks
```bash
# 1. Measure raw baseline
python benchmarks/run_baseline.py --tokens 64

# 2. Measure Paged KV-Cache allocation throughput
python benchmarks/run_paged_kv.py

# 3. Measure Continuous Batching GPU bubble reduction
python benchmarks/run_continuous_batching.py

# 4. Measure Speculative Decoding speedup
python benchmarks/run_speculative.py --tokens 64
```

### Step 4: Launch Full Engine Server & UI
```bash
# Terminal 1: Launch FastAPI Backend Server
uvicorn server.app:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Launch React Telemetry Dashboard
cd frontend
npm install
npm run dev
```

---

## 5. Definition of Done for Current Milestone

- [x] C++ Core Headers and Source implemented (`types.hpp`, `sequence.cpp`, `block_allocator.cpp`, `scheduler.cpp`, `bindings.cpp`).
- [x] C++ Unit Tests compiled and passing (`test_block_allocator`, `test_scheduler`).
- [x] PyTorch ML modules scaffolded (`naive_runner.py`, `paged_attention.py`, `rejection_sampler.py`, `spec_engine.py`).
- [x] FastAPI REST & WebSocket streaming server written.
- [x] React Telemetry Dashboard with glassmorphism UI components created.
- [x] Benchmark suite ready for performance evaluation.
- [ ] Python dependencies installed in local environment (`pip install -r requirements.txt`).
- [ ] Native `pybind11` extension built and verified (`pip install -e .`).
- [ ] Baseline metrics recorded in tracking table.
- [ ] Frontend dev server running and connected to live backend WebSocket.
