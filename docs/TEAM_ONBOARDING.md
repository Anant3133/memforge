# MemForge — Team Onboarding

**For:** anyone setting up MemForge for the first time
**Last verified:** 2026-10-25, on Windows 11 / Python 3.12.7 / Git Bash + PowerShell

Every command below was run top to bottom on a clean shell before this was
written. Where something goes wrong on this project, it is written down in
[§7 Troubleshooting](#7-when-it-goes-wrong) rather than left for you to
rediscover — including three failures that look like bugs and are not.

---

## 1. What you are setting up

MemForge is an LLM inference engine built from scratch: a C++ paged KV-cache
allocator and continuous-batching scheduler, a PyTorch execution layer, a
FastAPI server and a React dashboard. Four things have to work:

| Layer | What it needs |
| :--- | :--- |
| C++ core (`engine/`) | a C++17 compiler, to build a Python extension |
| Python runtime (`memforge_ml/`, `server/`) | Python 3.9+ and a virtualenv |
| Model weights | ~1 GB downloaded from Hugging Face, once |
| Dashboard (`frontend/`) | Node.js 18+ |

You can skip the dashboard if you are not working on it. You cannot skip the
C++ extension — the engine does not start without it.

---

## 2. Setup

### 2.1 Clone and enter

```bash
git clone https://github.com/Anant3133/memforge.git
cd memforge
```

### 2.2 Create and activate the virtualenv

The single most common mistake on this project is forgetting to activate it, so
it is worth checking every time you open a new terminal.

```bash
python -m venv .venv
```

```bash
source .venv/Scripts/activate      # Git Bash on Windows
source .venv/bin/activate          # macOS / Linux
```

```powershell
.\.venv\Scripts\Activate.ps1       # PowerShell
```

Your prompt should now start with `(.venv)`. If PowerShell refuses with an
execution-policy error, see [§7.6](#76-powershell-refuses-to-activate-the-venv).

### 2.3 Install the Python dependencies

```bash
pip install -r requirements.txt
```

This pulls PyTorch, which is a large download. Everything the project needs is
in that one file — `rich`, `psutil`, `httpx`, `websockets`, `matplotlib` and the
rest are already listed, so **you should not need to `pip install` anything
else**. If you think you do, ask Anant rather than adding it: `requirements.txt`
is a shared file and parallel edits conflict.

### 2.4 Build the C++ extension

```bash
python setup.py build_ext --inplace
```

This compiles `engine/src/*.cpp` into `memforge_c`, the native module the engine
imports. It takes 30-60 seconds. Check it worked:

```bash
python -c "import memforge_c; print(memforge_c.BlockAllocator(8, 16).get_total_blocks())"
# expect: 8
```

If it fails to compile, you are missing a C++ compiler — see
[§7.1](#71-the-c-extension-will-not-build).

### 2.5 Download the model weights

```bash
HF_HUB_DISABLE_SYMLINKS=1 hf download HuggingFaceTB/SmolLM2-360M-Instruct
HF_HUB_DISABLE_SYMLINKS=1 hf download HuggingFaceTB/SmolLM2-135M-Instruct
```

**`HF_HUB_DISABLE_SYMLINKS=1` is not optional on Windows.** The Hugging Face
cache normally stores one copy of each blob and symlinks to it, and creating a
symlink on Windows needs either Developer Mode or administrator rights. Without
the flag you get a confusing privilege error part-way through a 1 GB download.
The flag makes it copy instead, which costs disk space and always works.

The second model is the speculative-decoding draft model. It is optional — set
`MEMFORGE_LOAD_DRAFT=false` to skip loading it and save about 500 MB of RAM.

### 2.6 Run the environment doctor

```bash
python scripts/check_env.py
```

This checks all of the above and tells you exactly what to fix. **Run this
first whenever something is wrong** — it is faster than guessing. On a working
setup it reports 12 OK and exits 0. Warnings are safe to ignore if you know
why; failures are not.

### 2.7 Frontend (only if you are working on the dashboard)

```bash
cd frontend
npm install
npm run build        # confirms the toolchain works; takes ~8 s
cd ..
```

---

## 3. Verify the installation

### 3.1 Run the test suite

```bash
pytest tests/python/ -q
```

**Expect 157 passing in about 9 seconds.** These use a tiny randomly-initialised
model, so they need no weights and no network. If they pass, your Python and C++
layers are both working.

### 3.2 Run the C++ tests

```bash
python scripts/cpp_test_runner.py
```

Three suites: the block allocator, the scheduler, and the prefix cache. See
[§7.2](#72-the-c-test-runner-fails-with-permissionerror) if this fails — on some
machines it fails for reasons that are not the project's fault.

### 3.3 Start the server

```bash
uvicorn server.app:app --host 127.0.0.1 --port 8000
```

Then, in another terminal (with the venv activated):

```bash
curl http://127.0.0.1:8000/v1/health
```

**The first start takes 10-30 seconds** to load the model, and until it finishes
`/v1/health` reports `"engine_ready": false`. That is normal — wait for it.

Interactive API docs: **http://127.0.0.1:8000/docs**

### 3.4 Talk to it

```bash
python scripts/memforge_cli.py
```

A terminal chat client. Tokens stream live and each answer is followed by TTFT,
tokens/sec and block usage. `/help` lists the commands. This is the quickest way
to confirm the whole stack works end to end.

### 3.5 Open the dashboard

```bash
cd frontend && npm run dev
```

Then open the URL it prints (usually http://localhost:5173). It proxies `/v1`
and `/ws` to port 8000, so the server must be running.

---

## 4. A faster loop for day-to-day work

Loading the 360M model every time is slow. For iterating on the UI, the CLI or
tests, use the small model:

```bash
# FAST — starts in about 3 seconds
MEMFORGE_TARGET_MODEL=HuggingFaceTB/SmolLM2-135M-Instruct \
MEMFORGE_LOAD_DRAFT=false \
MEMFORGE_KV_BUDGET_MB=128 \
  uvicorn server.app:app --port 8000
```

And when you *want* things to break — testing saturation, preemption, 429s:

```bash
# PRESSURE — a deliberately tiny KV pool and a low admission limit
MEMFORGE_KV_BUDGET_MB=24 \
MEMFORGE_MAX_NUM_SEQS=2 \
MEMFORGE_MAX_CONCURRENCY=4 \
MEMFORGE_LOAD_DRAFT=false \
  uvicorn server.app:app --port 8000
```

**Environment variables are read at import time.** Setting one after the server
has started does nothing — you have to restart it. This catches everybody once.

The full list is in
[ASHUTOSH_SANSKAR_BHUMIKA_WORKPLAN.md](../ASHUTOSH_SANSKAR_BHUMIKA_WORKPLAN.md)
§3.5.

---

## 5. How we work

### 5.1 One person, one branch, one set of files

The project runs five parallel tracks with almost no merge pain, and the reason
is simple: **no two people edit the same file.** Ownership is in
[TEAM_EXECUTION_PLAN.md](../TEAM_EXECUTION_PLAN.md) §2.

```bash
git checkout main
git pull origin main
git checkout -b feat/<yourname>-<module>
```

If you need a change in a file you do not own, ask the owner. Not even a
one-line fix — that is how two branches end up editing the same line.

### 5.2 Before you commit

```bash
pytest tests/python/ -q               # must stay green
python scripts/cpp_test_runner.py     # if you touched C++
cd frontend && npm run build          # if you touched the frontend
```

A TypeScript error fails the frontend build, so `npm run build` is the real
check, not just that the dev server renders.

### 5.3 Push and open a PR

```bash
git add <your files>
git commit -m "feat(module): concise description"
git push -u origin feat/<yourname>-<module>
```

Rebase, never merge, when main moves:

```bash
git fetch origin
git rebase origin/main
```

One review approval. Squash or rebase merge, to keep history linear.

---

## 6. Where things are

```
engine/              C++ core: block allocator, scheduler, prefix cache
  include/memforge/  headers
  src/               implementation + pybind11 bindings
memforge_ml/         PyTorch execution layer
  runtime/           paged cache, continuous batching engine, prefix sharing
  models/            model loading, paged attention
  speculative/       draft proposal and rejection sampling
server/              FastAPI app
  api/               routes, websocket, disconnect handling
tests/python/        157 tests
tests/cpp/           three C++ suites
benchmarks/          benchmarks; results land in benchmarks/results/
scripts/             CLI, env doctor, traffic simulator, report exporter, pipeline
frontend/            React + Vite dashboard
docs/                this file and the API testing guide
```

Documents worth reading, in order:

1. [README.md](../README.md) — what the project is
2. [CURRENT_STATUS.md](../CURRENT_STATUS.md) — what works today
3. [PROGRESS_AND_NEXT_PHASES.md](../PROGRESS_AND_NEXT_PHASES.md) — measured
   results, known limitations, and what is next. The most useful document here
4. [TEAM_EXECUTION_PLAN.md](../TEAM_EXECUTION_PLAN.md) — who owns what
5. Your own track's plan: [ANANT_WORKPLAN.md](../ANANT_WORKPLAN.md) or
   [ASHUTOSH_SANSKAR_BHUMIKA_WORKPLAN.md](../ASHUTOSH_SANSKAR_BHUMIKA_WORKPLAN.md)
6. [docs/API_TESTING_GUIDE.md](API_TESTING_GUIDE.md) — every endpoint, with
   working examples

---

## 7. When it goes wrong

These are real failures seen on this project, not hypotheticals. Three of them
look like bugs and are not.

### 7.1 The C++ extension will not build

`python setup.py build_ext --inplace` fails, then `import memforge_c` raises
`ModuleNotFoundError`.

You need a C++17 compiler:

- **Windows:** install Visual Studio Build Tools with the "Desktop development
  with C++" workload. MSVC is what this project is verified against.
- **Linux:** `sudo apt install build-essential`
- **macOS:** `xcode-select --install`

`python scripts/check_env.py` tells you which compilers it can find.

### 7.2 The C++ test runner fails with PermissionError

```
PermissionError: [WinError 5] Access is denied
```

**This is environmental, not a project bug.** The runner builds each test into a
temporary directory and runs it from there, and some machines — antivirus,
corporate policy, certain sandboxes — refuse to execute a freshly written binary
out of `%TEMP%`. It reproduces on the pre-existing test suites too, so it is not
caused by any recent change.

Workaround: compile and run them directly.

```powershell
# PowerShell, from the repo root
& "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cl /nologo /EHsc /std:c++17 /O2 /Iengine\include `
   tests\cpp\test_prefix_cache.cpp engine\src\prefix_cache.cpp `
   engine\src\block_allocator.cpp engine\src\sequence.cpp `
   /Fobuild\cpptests\ /Febuild\cpptests\test_prefix_cache.exe
.\build\cpptests\test_prefix_cache.exe
```

If it happens to you, please say so — a second data point would help work out
how widespread it is.

### 7.3 `ld returned 116 exit status` from g++

**Also environmental.** The g++ linker is broken on some setups here and
produces a zero-byte executable. It fails identically on code that has never
been touched, so it is not a regression. `scripts/cpp_test_runner.py` tries g++
first and falls back to MSVC, which works. If you only have g++ and hit this,
install the MSVC Build Tools.

### 7.4 `pkill` does not stop the server on Windows

`pkill -f uvicorn` in Git Bash appears to succeed and the server keeps serving —
so you edit the code, "restart", and your change seems to have no effect. This
wasted a debugging session.

Kill it by the port instead:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen |
  Select-Object -ExpandProperty OwningProcess -Unique |
  ForEach-Object { Stop-Process -Id $_ -Force }
```

### 7.5 The server seems dead on startup

`/v1/health` returns `"engine_ready": false` and requests fail.

It is loading the model, which takes 10-30 seconds for the 360M target. Wait, or
poll:

```bash
until curl -s http://127.0.0.1:8000/v1/health | grep -q '"engine_ready":true'; do sleep 1; done
```

If `load_error` is non-null in the health response, that is a real failure — the
weights are probably not downloaded. See §2.5.

### 7.6 PowerShell refuses to activate the venv

```
Activate.ps1 cannot be loaded because running scripts is disabled on this system
```

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Or just use Git Bash with `source .venv/Scripts/activate`.

### 7.7 Tests pass but the server behaves differently

Almost always an environment variable. They are read **at import time**, so a
variable set after start-up has no effect, and one left over in your shell
affects every server you start from it. Check what the server actually thinks:

```bash
curl -s http://127.0.0.1:8000/v1/engine/status | python -m json.tool
```

The `config` block in that response is the truth.

### 7.8 "No module named memforge_ml" / "No module named server"

You are running from the wrong directory, or the venv is not active. Both are
fixed by running from the repo root with the venv activated. `pyproject.toml`
sets `pythonpath = ["."]` for pytest, but a bare `python somescript.py` needs
the working directory to be the repo root.

### 7.9 Everything is confusing

```bash
python scripts/check_env.py
```

Start there. It covers Python, the venv, packages, PyTorch, the native
extension, the compiler, RAM, CPU, model weights, disk and Node, and prints the
fix for whatever is wrong.

---

## 8. First tasks worth doing

Once you are set up, confirm you can actually work:

1. `pytest tests/python/ -q` — 157 passing
2. `python scripts/check_env.py` — 12 OK, exit 0
3. Start the server, then `python scripts/memforge_cli.py` and ask it something
4. `curl http://127.0.0.1:8000/metrics` — the Prometheus gauges
5. `python benchmarks/run_stress_test.py --spawn-server --requests 20` — should
   report 0 leaked blocks

If all five work, you have the whole stack running.

---

## 9. Found a gap in this document?

Then it is a bug in the document. Fix it or tell whoever owns `docs/` — the
guide is only useful while it still matches reality, and the person who just hit
the gap is the one who can see it clearly.
