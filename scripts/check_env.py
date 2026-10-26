#!/usr/bin/env python3
"""
MemForge environment doctor.

Answers one question: can this machine build and run MemForge, and if not, what
exactly is missing? Without it, a contributor with no C++ compiler gets an
`ImportError` on `memforge_c` and no idea why; with it they get the command to
run.

Design rules, because a diagnostic that misbehaves is worse than none:

  * **It never raises.** Every probe is wrapped; a failed check reports FAIL with
    the exception text rather than taking the whole report down. The person
    running this is already confused and does not need a traceback on top.
  * **Warnings are not failures.** Exit code is 0 unless something is genuinely
    broken, so CI can gate on it.
  * **No GPU is not a warning.** This is a CPU project. Reporting "WARN: no
    CUDA" on every run would train everyone to ignore the output.

Usage:
    python scripts/check_env.py
    python scripts/check_env.py --json       # machine-readable, for CI
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OK, WARN, FAIL = "OK", "WARN", "FAIL"

# Resolved lazily in probe_models(): this script must keep working when the
# project is not importable (no venv, extension unbuilt), which is exactly
# when someone runs it. Falls back to the same literals as
# memforge_ml/model_defaults.py.
TARGET_MODEL_ENV = "MEMFORGE_TARGET_MODEL"
DRAFT_MODEL_ENV = "MEMFORGE_DRAFT_MODEL"
TARGET_MODEL = os.environ.get(TARGET_MODEL_ENV, "HuggingFaceTB/SmolLM2-360M-Instruct")
DRAFT_MODEL = os.environ.get(DRAFT_MODEL_ENV, "HuggingFaceTB/SmolLM2-135M-Instruct")

# Packages whose absence breaks something concrete, with what it breaks.
REQUIRED_PACKAGES = {
    "torch": "the model execution layer",
    "transformers": "model loading",
    "fastapi": "the server",
    "uvicorn": "serving",
    "pydantic": "request validation",
    "pytest": "the test suite",
    "rich": "this script's output",
    "httpx": "the traffic simulator and CLI",
    "websockets": "WebSocket streaming",
    "psutil": "the stress harness",
    "numpy": "tensor interop",
    "pybind11": "building the C++ extension",
}
OPTIONAL_PACKAGES = {
    "matplotlib": "charts in the benchmark report exporter",
    "accelerate": "faster/low-memory model loading",
    "tqdm": "progress bars",
}


@dataclass
class Result:
    name: str
    status: str
    detail: str = ""
    fix: str = ""


@dataclass
class Report:
    results: List[Result] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "", fix: str = "") -> None:
        self.results.append(Result(name, status, detail, fix))

    def count(self, status: str) -> int:
        return sum(1 for r in self.results if r.status == status)

    @property
    def failed(self) -> bool:
        return self.count(FAIL) > 0


def check(report: Report, name: str) -> Callable:
    """
    Decorator that runs one probe and converts any exception into a FAIL.

    This is the "never raises" rule in one place rather than a try/except in
    every probe.
    """

    def wrapper(fn: Callable[[], Result]) -> None:
        try:
            result = fn()
            report.results.append(Result(name, result.status, result.detail, result.fix))
        except Exception as exc:  # noqa: BLE001 - deliberate catch-all
            report.add(name, FAIL, f"probe raised {type(exc).__name__}: {exc}")

    return wrapper


# ---------------------------------------------------------------- the probes

def probe_python() -> Result:
    v = sys.version_info
    detail = f"{v.major}.{v.minor}.{v.micro} ({platform.python_implementation()})"
    if (v.major, v.minor) < (3, 9):
        return Result("", FAIL, detail, "MemForge requires Python >= 3.9 (see pyproject.toml)")
    return Result("", OK, detail)


def probe_virtualenv() -> Result:
    active = sys.prefix != sys.base_prefix
    if active:
        return Result("", OK, f"active ({Path(sys.prefix).name})")
    return Result(
        "", WARN, "not active - packages may be missing or come from the system Python",
        "source .venv/Scripts/activate   (Git Bash)\n"
        "        .\\.venv\\Scripts\\Activate.ps1   (PowerShell)",
    )


def probe_torch() -> Result:
    try:
        import torch
    except ImportError as exc:
        return Result("", FAIL, str(exc), "pip install -r requirements.txt")

    bits = [f"{torch.__version__}"]
    if torch.cuda.is_available():
        try:
            bits.append(f"CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}")
        except Exception:
            bits.append("CUDA available")
    else:
        # Deliberately OK, not WARN: MemForge is a CPU project and every
        # measured benchmark in the repo is CPU-only.
        bits.append("CPU only - expected for this project")
    return Result("", OK, ", ".join(bits))


def probe_native_extension() -> Result:
    try:
        import memforge_c
    except ImportError as exc:
        return Result(
            "", FAIL, str(exc),
            "python setup.py build_ext --inplace    (needs a C++17 compiler)",
        )

    try:
        allocator = memforge_c.BlockAllocator(8, 16)
        has_prefix = hasattr(memforge_c, "PrefixCache")
        detail = (
            f"importable, {allocator.get_total_blocks()} test blocks, "
            f"PrefixCache {'present' if has_prefix else 'MISSING'}"
        )
        if not has_prefix:
            return Result(
                "", WARN, detail,
                "the extension predates prefix sharing; rebuild with "
                "python setup.py build_ext --inplace",
            )
        return Result("", OK, detail)
    except Exception as exc:
        return Result("", FAIL, f"imported but unusable: {exc}",
                      "python setup.py build_ext --inplace")


def probe_compiler() -> Result:
    found = [name for name in ("cl", "g++", "clang++") if shutil.which(name)]
    if not found:
        fix = {
            "Windows": "install Visual Studio Build Tools with the C++ workload",
            "Linux": "sudo apt install build-essential",
            "Darwin": "xcode-select --install",
        }.get(platform.system(), "install a C++17 compiler")
        return Result("", FAIL, "no C++ compiler on PATH", fix)

    detail = ", ".join(found)
    # Note the known-broken linker here rather than only in the docs: it is the
    # single most confusing failure on this project.
    if found == ["g++"] and platform.system() == "Windows":
        return Result(
            "", WARN, f"{detail} (only g++)",
            "g++ cannot link on some Windows setups (ld returns 116, zero-byte "
            "output). scripts/cpp_test_runner.py falls back to MSVC, so install "
            "VS Build Tools if the C++ tests fail to link.",
        )
    return Result("", OK, detail)


def probe_memory() -> Result:
    try:
        import psutil
    except ImportError:
        return Result("", WARN, "psutil not installed, cannot check RAM",
                      "pip install -r requirements.txt")

    mem = psutil.virtual_memory()
    total_gb = mem.total / 1024**3
    avail_gb = mem.available / 1024**3
    detail = f"{total_gb:.1f} GB total, {avail_gb:.1f} GB available"

    # The 360M target in fp32 plus its KV pool needs roughly 2.5 GB to be
    # comfortable; the 135M draft adds about 0.5 GB.
    if avail_gb < 2.0:
        return Result(
            "", WARN, detail,
            "under 2 GB free: use MEMFORGE_TARGET_MODEL="
            f"{DRAFT_MODEL} and MEMFORGE_LOAD_DRAFT=false",
        )
    return Result("", OK, detail)


def probe_cpu() -> Result:
    count = os.cpu_count() or 0
    bits = [f"{count} CPUs"]

    flags = _cpu_flags()
    if flags:
        simd = [f for f in ("avx512f", "avx2", "avx", "sse4_2") if f in flags]
        bits.append(", ".join(simd).upper() if simd else "no AVX detected")
    else:
        bits.append("SIMD flags undetectable on this platform")

    machine = platform.machine()
    if machine:
        bits.append(machine)
    return Result("", OK, " | ".join(bits))


def _cpu_flags() -> set:
    """CPU feature flags, best effort. Returns an empty set when unavailable."""
    system = platform.system()
    try:
        if system == "Linux":
            text = Path("/proc/cpuinfo").read_text(errors="ignore")
            for line in text.splitlines():
                if line.startswith("flags") or line.startswith("Features"):
                    return set(line.split(":", 1)[1].split())
        elif system == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.features", "machdep.cpu.leaf7_features"],
                capture_output=True, text=True, timeout=5,
            )
            return {f.lower() for f in out.stdout.split()}
        elif system == "Windows":
            # No cpuinfo file. numpy exposes what its own build detected, which
            # is the best answer available without a native dependency. The
            # attribute moved from numpy.core to numpy._core in numpy 2.x, and
            # touching the old name emits a DeprecationWarning, so try the new
            # location first rather than catching the warning afterwards.
            import numpy
            opts = {}
            for module_name in ("numpy._core._multiarray_umath",
                                "numpy.core._multiarray_umath"):
                try:
                    module = importlib.import_module(module_name)
                except Exception:
                    continue
                opts = getattr(module, "__cpu_features__", {}) or {}
                if opts:
                    break
            if not opts:
                opts = getattr(numpy, "__cpu_features__", {}) or {}
            return {k.lower() for k, v in opts.items() if v}
    except Exception:
        pass
    return set()


def _hf_cache_dir() -> Path:
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(var):
            return Path(os.environ[var])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _model_cached(repo_id: str) -> bool:
    folder = "models--" + repo_id.replace("/", "--")
    path = _hf_cache_dir() / folder
    if not path.is_dir():
        return False
    # A directory can exist from an interrupted download, so require an actual
    # snapshot with files in it.
    snapshots = path / "snapshots"
    if not snapshots.is_dir():
        return False
    return any(any(s.iterdir()) for s in snapshots.iterdir() if s.is_dir())


def probe_models() -> Result:
    cache = _hf_cache_dir()
    target = _model_cached(TARGET_MODEL)
    draft = _model_cached(DRAFT_MODEL)

    if target and draft:
        return Result("", OK, f"target + draft cached in {cache}")

    missing = [m for m, present in ((TARGET_MODEL, target), (DRAFT_MODEL, draft))
               if not present]
    fix = "\n".join(
        f"HF_HUB_DISABLE_SYMLINKS=1 hf download {m}" for m in missing
    ) + "\n        (HF_HUB_DISABLE_SYMLINKS=1 avoids a Windows symlink-privilege error)"

    # The target missing is worse than the draft missing: the draft is optional
    # and the server degrades gracefully without it.
    status = FAIL if not target else WARN
    detail = "missing: " + ", ".join(missing)
    return Result("", status, detail, fix)


def probe_disk() -> Result:
    cache = _hf_cache_dir()
    probe_path = cache if cache.exists() else Path.home()
    usage = shutil.disk_usage(probe_path)
    free_gb = usage.free / 1024**3
    detail = f"{free_gb:.1f} GB free on the drive holding {probe_path}"
    if free_gb < 2.0:
        return Result("", WARN, detail, "model weights need about 1 GB")
    return Result("", OK, detail)


def probe_packages() -> Result:
    missing, versions = [], []
    for name in REQUIRED_PACKAGES:
        try:
            versions.append(f"{name} {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            missing.append(name)

    if missing:
        broken = ", ".join(f"{m} ({REQUIRED_PACKAGES[m]})" for m in missing)
        return Result("", FAIL, f"missing: {broken}", "pip install -r requirements.txt")
    return Result("", OK, f"all {len(REQUIRED_PACKAGES)} required packages present")


def probe_optional_packages() -> Result:
    missing = []
    for name in OPTIONAL_PACKAGES:
        try:
            metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
    if missing:
        detail = "missing: " + ", ".join(f"{m} ({OPTIONAL_PACKAGES[m]})" for m in missing)
        return Result("", WARN, detail, "pip install -r requirements.txt")
    return Result("", OK, f"all {len(OPTIONAL_PACKAGES)} optional packages present")


def probe_project_layout() -> Result:
    expected = [
        "engine/src/block_allocator.cpp",
        "engine/src/prefix_cache.cpp",
        "memforge_ml/runtime/continuous_engine.py",
        "server/app.py",
        "tests/python",
        "requirements.txt",
    ]
    missing = [p for p in expected if not (PROJECT_ROOT / p).exists()]
    if missing:
        return Result(
            "", FAIL, "missing: " + ", ".join(missing),
            f"run this from the repo root (detected {PROJECT_ROOT})",
        )
    return Result("", OK, f"repo root {PROJECT_ROOT}")


def probe_frontend() -> Result:
    frontend = PROJECT_ROOT / "frontend"
    if not frontend.is_dir():
        return Result("", WARN, "frontend/ not found", "")
    if not (frontend / "node_modules").is_dir():
        return Result("", WARN, "node_modules not installed",
                      "cd frontend && npm install")
    node = shutil.which("node")
    if not node:
        return Result("", WARN, "node_modules present but node is not on PATH",
                      "install Node.js 18+ to run the dashboard")
    try:
        version = subprocess.run([node, "--version"], capture_output=True,
                                 text=True, timeout=5).stdout.strip()
    except Exception:
        version = "unknown version"
    return Result("", OK, f"node {version}, node_modules installed")


# ----------------------------------------------------------------- reporting

def run_all() -> Report:
    report = Report()
    probes = [
        ("Python version", probe_python),
        ("Virtualenv", probe_virtualenv),
        ("Project layout", probe_project_layout),
        ("Required packages", probe_packages),
        ("Optional packages", probe_optional_packages),
        ("PyTorch", probe_torch),
        ("Native extension (memforge_c)", probe_native_extension),
        ("C++ compiler", probe_compiler),
        ("System memory", probe_memory),
        ("CPU", probe_cpu),
        ("Model weights", probe_models),
        ("Disk space", probe_disk),
        ("Frontend toolchain", probe_frontend),
    ]
    for name, fn in probes:
        check(report, name)(fn)
    return report


def render_rich(report: Report) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()
    styles = {OK: "bold green", WARN: "bold yellow", FAIL: "bold red"}

    table = Table(show_header=True, header_style="bold cyan", expand=True)
    table.add_column("", width=6)
    table.add_column("Check", width=30)
    table.add_column("Detail", overflow="fold")

    for r in report.results:
        detail = r.detail
        if r.fix:
            detail = f"{detail}\n[dim]-> {r.fix}[/dim]" if detail else f"[dim]-> {r.fix}[/dim]"
        table.add_row(f"[{styles[r.status]}]{r.status}[/{styles[r.status]}]", r.name, detail)

    console.print(Panel.fit("[bold cyan]MemForge Environment Doctor[/bold cyan]",
                            border_style="cyan"))
    console.print(table)

    n_ok, n_warn, n_fail = report.count(OK), report.count(WARN), report.count(FAIL)
    summary = (f"[bold green]{n_ok} OK[/bold green]  |  "
               f"[bold yellow]{n_warn} warning(s)[/bold yellow]  |  "
               f"[bold red]{n_fail} failure(s)[/bold red]")
    if n_fail:
        verdict = "[bold red]Environment is not ready.[/bold red] Fix the failures above."
    elif n_warn:
        verdict = "[bold yellow]Usable, with caveats.[/bold yellow] Warnings are safe to ignore if you know why."
    else:
        verdict = "[bold green]Environment is ready.[/bold green]"
    console.print(Panel(f"{summary}\n{verdict}", border_style="cyan"))


def render_plain(report: Report) -> None:
    """Fallback when rich is the thing that is missing."""
    print("MemForge Environment Doctor")
    print("-" * 70)
    for r in report.results:
        print(f"  {r.status:5} {r.name:30} {r.detail}")
        if r.fix:
            for line in r.fix.splitlines():
                print(f"        -> {line.strip()}")
    print("-" * 70)
    print(f"  {report.count(OK)} OK, {report.count(WARN)} warning(s), "
          f"{report.count(FAIL)} failure(s)")


def main() -> int:
    parser = argparse.ArgumentParser(description="MemForge environment diagnostics")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    report = run_all()

    if args.json:
        print(json.dumps({
            "ok": not report.failed,
            "counts": {s: report.count(s) for s in (OK, WARN, FAIL)},
            "checks": [
                {"name": r.name, "status": r.status, "detail": r.detail, "fix": r.fix}
                for r in report.results
            ],
        }, indent=2))
    else:
        try:
            render_rich(report)
        except Exception:
            # rich missing or a terminal it cannot drive: still print something.
            render_plain(report)

    # Warnings do not fail the run, so CI can gate on this without being noisy.
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
