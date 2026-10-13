"""
Build and run the C++ unit tests with whichever toolchain actually works.

g++ is tried first because it needs no setup, but a broken or sandboxed linker
is common on Windows (it shows up as `ld returned <n> exit status` with a
zero-byte executable). MSVC is tried next, since the pybind11 extension is
already built with it, so a working MSVC is a safe assumption on Windows.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VSWHERE = Path(
    os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"

FALLBACK_VCVARS = [
    r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
    r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
    r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
]


@dataclass
class BuildResult:
    ok: bool
    toolchain: str
    output: str
    detail: str = ""


def find_vcvars() -> Optional[Path]:
    """Locate vcvars64.bat, preferring whatever vswhere reports."""
    if VSWHERE.exists():
        try:
            completed = subprocess.run(
                [
                    str(VSWHERE),
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            root = completed.stdout.strip().splitlines()
            if root:
                candidate = Path(root[0]) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                if candidate.exists():
                    return candidate
        except (OSError, subprocess.SubprocessError):
            pass

    for path in FALLBACK_VCVARS:
        if Path(path).exists():
            return Path(path)
    return None


def _run_exe(exe: Path) -> BuildResult:
    completed = subprocess.run(
        [str(exe)], capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=300
    )
    combined = (completed.stdout or "") + (completed.stderr or "")
    return BuildResult(
        ok=completed.returncode == 0,
        toolchain="",
        output=combined,
        detail="" if completed.returncode == 0 else "test assertions failed",
    )


def _try_gpp(sources: Sequence[str], exe: Path) -> Optional[BuildResult]:
    """Build with g++. Returns None if g++ is unusable, so the caller falls back."""
    if shutil.which("g++") is None:
        return None

    command = [
        "g++",
        "-std=c++17",
        "-O2",
        "-I",
        "engine/include",
        *sources,
        "-o",
        str(exe),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=600
    )
    # A broken linker reports failure, or silently emits an empty binary.
    if completed.returncode != 0 or not exe.exists() or exe.stat().st_size == 0:
        return None
    return BuildResult(ok=True, toolchain="g++", output=completed.stdout)


def _try_msvc(sources: Sequence[str], exe: Path) -> Optional[BuildResult]:
    """Build with MSVC through vcvars64.bat."""
    if sys.platform != "win32":
        return None
    vcvars = find_vcvars()
    if vcvars is None:
        return None

    obj_dir = Path(tempfile.mkdtemp(prefix="memforge_cpp_obj_"))
    source_args = " ".join(s.replace("/", "\\") for s in sources)
    script = (
        "@echo off\r\n"
        f'call "{vcvars}" >nul 2>&1\r\n'
        f'cd /d "{PROJECT_ROOT}"\r\n'
        # /Fo must keep its trailing backslash to mean "directory", and that
        # backslash would escape a closing quote, so the path is left unquoted.
        # Temp dirs created here contain no spaces.
        "cl /nologo /EHsc /std:c++17 /O2 /I engine\\include "
        f'{source_args} /Fe:"{exe}" /Fo:{obj_dir}\\ \r\n'
        "exit /b %ERRORLEVEL%\r\n"
    )
    bat = obj_dir / "build.bat"
    bat.write_text(script, encoding="ascii")

    completed = subprocess.run(
        ["cmd", "/c", str(bat)], capture_output=True, text=True, timeout=900
    )
    shutil.rmtree(obj_dir, ignore_errors=True)

    if completed.returncode != 0 or not exe.exists() or exe.stat().st_size == 0:
        return BuildResult(
            ok=False,
            toolchain="MSVC",
            output=(completed.stdout or "") + (completed.stderr or ""),
            detail="MSVC compilation failed",
        )
    return BuildResult(ok=True, toolchain="MSVC", output=completed.stdout)


def _compile_only_check(sources: Sequence[str]) -> bool:
    """Verify the sources at least compile, when no toolchain can link."""
    if shutil.which("g++") is None:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        for source in sources:
            obj = Path(tmp) / (Path(source).stem + ".o")
            completed = subprocess.run(
                ["g++", "-std=c++17", "-c", "-I", "engine/include", source, "-o", str(obj)],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
                timeout=600,
            )
            if completed.returncode != 0:
                return False
    return True


def build_and_run(name: str, sources: List[str]) -> BuildResult:
    """
    Build `sources` into a test executable and run it.

    Tries g++, then MSVC. If neither can link, falls back to a compile-only
    check so a broken local linker is reported as an environment problem rather
    than a code failure.
    """
    exe_dir = Path(tempfile.mkdtemp(prefix="memforge_cpp_bin_"))
    exe = exe_dir / f"{name}.exe"
    try:
        for attempt in (_try_gpp, _try_msvc):
            built = attempt(sources, exe)
            if built is None:
                continue
            if not built.ok:
                return built
            result = _run_exe(exe)
            result.toolchain = built.toolchain
            return result

        compiles = _compile_only_check(sources)
        return BuildResult(
            ok=False,
            toolchain="none",
            output="",
            detail=(
                "sources compile but no working linker was found"
                if compiles
                else "no usable C++ toolchain found"
            ),
        )
    finally:
        shutil.rmtree(exe_dir, ignore_errors=True)


CPP_TEST_SUITES = {
    "test_block_allocator": [
        "tests/cpp/test_block_allocator.cpp",
        "engine/src/block_allocator.cpp",
        "engine/src/sequence.cpp",
    ],
    "test_scheduler": [
        "tests/cpp/test_scheduler.cpp",
        "engine/src/scheduler.cpp",
        "engine/src/block_allocator.cpp",
        "engine/src/sequence.cpp",
    ],
}


if __name__ == "__main__":
    exit_code = 0
    for suite, sources in CPP_TEST_SUITES.items():
        result = build_and_run(suite, sources)
        status = "PASS" if result.ok else "FAIL"
        print(f"[{status}] {suite} (toolchain: {result.toolchain or 'n/a'})")
        if result.output.strip():
            print("    " + result.output.strip().replace("\n", "\n    "))
        if not result.ok:
            print(f"    detail: {result.detail}")
            exit_code = 1
    raise SystemExit(exit_code)
