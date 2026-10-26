#!/usr/bin/env python3
"""
Run the C++ core under a memory sanitizer, as the last open Phase F item.

Why two tools rather than one. `benchmarks/run_stress_test.py` already shows
flat RSS and zero leaked blocks under sustained load, but it measures from
Python and so can only see *logical* leaks -- blocks the allocator never takes
back. It is blind to memory misuse inside the C++ itself: a use-after-free on a
recycled block, a read past the end of a block table, a double free. Those
corrupt silently, and the symptom is wrong KV, which produces fluent and
confident wrong text rather than a crash.

What it runs depends on the platform, and the difference is worth knowing:

  * **Linux / macOS** -- one pass with `-fsanitize=address`. gcc and clang ship
    LeakSanitizer alongside it, so leaks, use-after-free and buffer overflows
    are all covered at once.
  * **Windows (MSVC)** -- two passes, because MSVC's AddressSanitizer has **no
    leak detection**; it ships no LeakSanitizer. So `/fsanitize=address` covers
    the memory errors and a second build against the debug CRT with
    `_CRTDBG_LEAK_CHECK_DF` covers leaks.

The two-pass arrangement is a workaround for the weaker platform, not the
intended design.

None of that is taken on trust. `--verify-tooling` compiles
`tests/cpp/sanitizer_probe.cpp`, which holds a known leak, a known
use-after-free and a known overflow, and reports which defect classes each pass
actually caught. A sanitizer that silently detects nothing looks exactly like
clean code, so the capability check is part of the result rather than a
footnote -- and it has already caught two configuration faults that would
otherwise have read as success.

Usage (identical on every platform):
    python scripts/cpp_asan.py
    python scripts/cpp_asan.py --cycles 100 --requests 16
    python scripts/cpp_asan.py --verify-tooling
"""

from __future__ import annotations

import argparse
import glob
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cpp_test_runner import find_vcvars  # noqa: E402

DRIVER = "tests/cpp/asan_sustained_load.cpp"
ENGINE_SOURCES = [
    "engine/src/block_allocator.cpp",
    "engine/src/prefix_cache.cpp",
    "engine/src/scheduler.cpp",
    "engine/src/sequence.cpp",
]

ASAN_DLL_GLOB = (
    "C:/Program Files*/Microsoft Visual Studio/*/*/VC/Tools/MSVC/*/bin/"
    "Hostx64/x64/clang_rt.asan_dynamic-x86_64.dll"
)

# Markers in sanitizer output, mapped to the defect class they indicate.
MARKERS = {
    "heap-use-after-free": "use-after-free",
    "heap-buffer-overflow": "buffer-overflow",
    "attempting double-free": "double-free",
    "detected memory leaks": "leak",
    "Detected memory leaks": "leak",
}


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def find_asan_dll() -> Optional[Path]:
    """
    Locate the ASan runtime DLL.

    It lives in the MSVC bin directory, which is not on PATH outside a vcvars
    shell. Without it the process exits with 0xC0000135 before `main` runs --
    which prints nothing and looks exactly like a clean sanitizer run, so this
    is copied next to the executable rather than relied on from PATH.
    """
    matches = sorted(glob.glob(ASAN_DLL_GLOB))
    return Path(matches[-1]) if matches else None


def msvc_build(
    sources: List[str],
    exe: Path,
    obj_dir: Path,
    extra_flags: str,
    defines: str = "",
) -> Tuple[bool, str]:
    """Compile `sources` with MSVC inside a vcvars shell."""
    vcvars = find_vcvars()
    if vcvars is None:
        return False, "vcvars64.bat not found; MSVC is required for this script"

    source_args = " ".join(str(s).replace("/", "\\") for s in sources)
    script = (
        "@echo off\r\n"
        f'call "{vcvars}" >nul 2>&1\r\n'
        f'cd /d "{PROJECT_ROOT}"\r\n'
        # /Fo keeps its trailing backslash to mean "directory", and that
        # backslash would escape a closing quote, so the path stays unquoted.
        f"cl /nologo /EHsc /std:c++17 /Zi {extra_flags} {defines} "
        f'/I engine\\include {source_args} /Fe:"{exe}" /Fo:{obj_dir}\\ \r\n'
        "exit /b %ERRORLEVEL%\r\n"
    )
    bat = obj_dir / "build.bat"
    bat.write_text(script, encoding="ascii")

    done = subprocess.run(
        ["cmd", "/c", str(bat)], capture_output=True, text=True, timeout=1800
    )
    output = (done.stdout or "") + (done.stderr or "")
    if done.returncode != 0 or not exe.exists() or exe.stat().st_size == 0:
        return False, output
    return True, output


def posix_build(
    sources: List[str],
    exe: Path,
    extra_flags: List[str],
    defines: List[str],
) -> Tuple[bool, str]:
    """
    Compile with clang++ or g++ on Linux/macOS.

    clang++ is preferred where both exist: its AddressSanitizer diagnostics name
    the allocation site as well as the fault, which is the difference between
    "something freed this twice" and "this line freed it".
    """
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        return False, "neither clang++ nor g++ is on PATH"

    command = [
        compiler, "-std=c++17", "-g", "-O1",
        *extra_flags, *defines,
        "-I", "engine/include",
        *sources, "-o", str(exe),
    ]
    done = subprocess.run(
        command, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=1800
    )
    output = (done.stdout or "") + (done.stderr or "")
    if done.returncode != 0 or not exe.exists() or exe.stat().st_size == 0:
        return False, f"$ {' '.join(command)}\n{output}"
    return True, output


def build(
    sources: List[str],
    exe: Path,
    obj_dir: Path,
    msvc_flags: str,
    posix_flags: List[str],
    msvc_defines: str = "",
    posix_defines: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """Dispatch to whichever toolchain this platform has."""
    if sys.platform == "win32":
        return msvc_build(sources, exe, obj_dir, msvc_flags, msvc_defines)
    return posix_build(sources, exe, posix_flags, posix_defines or [])


def run_exe(exe: Path, args: List[str], cwd: Path) -> Tuple[int, str]:
    """
    Run a freshly built executable.

    The retry is not superstition: on Windows a just-written binary can be
    briefly locked by the on-access virus scanner, and `CreateProcess` then
    fails with "Access is denied". Without the retry that surfaces as a crash
    in the middle of a sanitizer run and looks like a tooling failure.
    """
    last: Optional[OSError] = None
    for attempt in range(5):
        try:
            done = subprocess.run(
                [str(exe), *args], capture_output=True, text=True,
                cwd=str(cwd), timeout=3600,
            )
            return done.returncode, (done.stdout or "") + (done.stderr or "")
        except PermissionError as exc:  # scanner still holding the file
            last = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"could not launch {exe}: {last}")


def classify(output: str) -> List[str]:
    """Which defect classes the sanitizer reported."""
    found = []
    for marker, kind in MARKERS.items():
        if marker in output and kind not in found:
            found.append(kind)
    return found


def sanitizer_passes() -> List[dict]:
    """
    The passes to run on this platform.

    On Linux/macOS one pass suffices: `-fsanitize=address` there includes
    LeakSanitizer. On Windows it does not, so leaks need a separate debug-CRT
    build. Keeping this in one function is what stops `--verify-tooling` from
    certifying a different configuration than the one that actually runs.
    """
    if sys.platform == "win32":
        return [
            {
                "name": "AddressSanitizer",
                "msvc_flags": "/fsanitize=address /MD /Od",
                "msvc_defines": "",
                "posix_flags": [],
                "posix_defines": [],
                "needs_dll": True,
            },
            {
                "name": "CRT debug heap",
                "msvc_flags": "/MDd /Od",
                "msvc_defines": "/DMEMFORGE_CRT_LEAK_CHECK",
                "posix_flags": [],
                "posix_defines": [],
                "needs_dll": False,
            },
        ]
    return [
        {
            "name": "AddressSanitizer + LeakSanitizer",
            "msvc_flags": "",
            "msvc_defines": "",
            "posix_flags": ["-fsanitize=address", "-fno-omit-frame-pointer"],
            "posix_defines": [],
            "needs_dll": False,
        }
    ]


def verify_tooling() -> int:
    """
    Establish what each pass can actually detect, using known defects.

    Without this the script's clean result is unfalsifiable: a sanitizer that
    detects nothing is indistinguishable from code with nothing to detect.
    """
    banner("Tooling capability check")
    probe = PROJECT_ROOT / "tests" / "cpp" / "sanitizer_probe.cpp"
    if not probe.exists():
        print(f"missing {probe.relative_to(PROJECT_ROOT)}")
        return 1

    expected = ["leak", "use-after-free", "buffer-overflow"]
    table: Dict[str, Dict[str, bool]] = {}

    for spec in sanitizer_passes():
        name = spec["name"]
        needs_dll = spec["needs_dll"]
        obj = Path(tempfile.mkdtemp(prefix="memforge_probe_"))
        exe = obj / ("probe.exe" if sys.platform == "win32" else "probe")
        ok, output = build(
            [str(probe)], exe, obj,
            spec["msvc_flags"], spec["posix_flags"],
            spec["msvc_defines"], spec["posix_defines"],
        )
        if not ok:
            print(f"  {name}: BUILD FAILED")
            print("   " + output.strip().replace("\n", "\n   ")[:800])
            shutil.rmtree(obj, ignore_errors=True)
            table[name] = {k: False for k in expected}
            continue
        if needs_dll:
            dll = find_asan_dll()
            if dll is not None:
                shutil.copy2(dll, obj / dll.name)

        row = {}
        for mode in expected:
            arg = {"leak": "leak", "use-after-free": "uaf",
                   "buffer-overflow": "overflow"}[mode]
            _, out = run_exe(exe, [arg], obj)
            row[mode] = mode in classify(out)
        table[name] = row
        shutil.rmtree(obj, ignore_errors=True)

    width = max(len(n) for n in table) + 2
    print(f"\n  {'pass'.ljust(width)}" + "".join(m.ljust(20) for m in expected))
    for name, row in table.items():
        cells = "".join(("detected" if row[m] else "MISSED").ljust(20) for m in expected)
        print(f"  {name.ljust(width)}{cells}")

    print()
    asan = table.get("AddressSanitizer", {})
    if sys.platform == "win32" and not asan.get("leak", False):
        print("  AddressSanitizer misses leaks on this toolchain, as expected:")
        print("  MSVC ships no LeakSanitizer. This is why the CRT debug-heap")
        print("  pass exists, and why running ASan alone would not close the")
        print("  Phase F item.")
    elif sys.platform != "win32":
        print("  One pass covers everything here: gcc/clang ship LeakSanitizer")
        print("  with AddressSanitizer, so the second Windows-only pass is")
        print("  unnecessary on this platform.")
    covered = {m for row in table.values() for m, hit in row.items() if hit}
    missing = [m for m in expected if m not in covered]
    if missing:
        print(f"  NOT COVERED BY ANY PASS: {', '.join(missing)}")
        print("  Results below cannot be read as evidence for those classes.")
        return 1
    print(f"  All of {', '.join(expected)} are covered by at least one pass.")
    return 0


def run_pass(spec: dict, cycles: int, requests: int) -> Tuple[bool, List[str]]:
    name = spec["name"]
    needs_dll = spec["needs_dll"]
    banner(f"{name}: sustained load, {cycles} cycles x {requests} requests")
    obj = Path(tempfile.mkdtemp(prefix="memforge_asan_"))
    exe = obj / ("sustained_load.exe" if sys.platform == "win32" else "sustained_load")
    try:
        ok, output = build(
            [DRIVER, *ENGINE_SOURCES], exe, obj,
            spec["msvc_flags"], spec["posix_flags"],
            spec["msvc_defines"], spec["posix_defines"],
        )
        if not ok:
            print("  BUILD FAILED")
            print("   " + output.strip().replace("\n", "\n   ")[:2000])
            return False, []
        if needs_dll:
            dll = find_asan_dll()
            if dll is None:
                print("  ASan runtime DLL not found; the run would die at startup")
                return False, []
            shutil.copy2(dll, obj / dll.name)

        code, out = run_exe(exe, [str(cycles), str(requests)], obj)
        defects = classify(out)
        print("   " + out.strip().replace("\n", "\n   ")[:4000])
        clean = code == 0 and not defects
        print(f"\n  exit={code}  defects={defects or 'none'}  "
              f"-> {'CLEAN' if clean else 'PROBLEM'}")
        return clean, defects
    finally:
        shutil.rmtree(obj, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=40)
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument(
        "--verify-tooling",
        action="store_true",
        help="only check what the sanitizers can detect, then exit",
    )
    args = parser.parse_args()

    tooling = verify_tooling()
    if args.verify_tooling:
        return tooling
    if tooling != 0:
        print("\nRefusing to report a clean result with unverified tooling.")
        return 1

    results = []
    for spec in sanitizer_passes():
        clean, defects = run_pass(spec, args.cycles, args.requests)
        results.append((spec["name"], clean, defects))

    banner("Verdict")
    for name, clean, defects in results:
        status = "clean" if clean else f"PROBLEM: {defects or 'see output'}"
        print(f"  {name:20} {status}")

    all_clean = all(clean for _, clean, _ in results)
    if all_clean:
        print(
            f"\n  No use-after-free, buffer overflow, double free or heap leak "
            f"across {args.cycles} cycles of {args.requests} concurrent "
            f"requests, with prefix sharing, copy-on-write, preemption and "
            f"cache eviction all exercised. The driver also asserts every "
            f"cycle that the free list returns to its starting size, which is "
            f"the logical-leak half that no sanitizer checks."
        )
    return 0 if all_clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
