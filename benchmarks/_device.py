"""
Shared device and dtype resolution for the benchmark harnesses.

Four of the benchmarks hardcoded `torch.float32` and implicitly ran on CPU, so
none of them could produce a GPU number even once a CUDA build was installed --
which is what blocked Phase E from being measurable at all. This is the one
place that decides where a benchmark runs, so the decision cannot drift between
scripts and a fix lands everywhere.

**Explicit requests are never silently downgraded.** Asking for `--device cuda`
on a machine without CUDA is an error, not a fallback to CPU. A silent fallback
is the worst possible behaviour here: the run succeeds, the JSON says `cuda`
because that is what was asked for, and the numbers in the report are CPU
numbers wearing a GPU label. Every other guard in this project exists to stop
exactly that kind of confident-but-wrong result, and a benchmark harness is no
place to break the pattern.

`auto` is the default and does the sensible thing: CUDA if it is really there,
fp16 on GPU and fp32 on CPU, because fp16 on CPU has no hardware support and
measures the emulation rather than the engine.
"""

from __future__ import annotations

import argparse
import platform
from typing import Any, Dict, Tuple

import torch

#: Names accepted by `--dtype`, mapped to the torch dtype.
DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def add_device_args(parser: argparse.ArgumentParser) -> None:
    """Add `--device` and `--dtype` to a benchmark's parser."""
    parser.add_argument(
        "--device",
        default="auto",
        help="auto (default: cuda when available), cuda, cuda:N, or cpu. An "
             "explicit cuda request fails rather than falling back to CPU.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", *DTYPES.keys()],
        help="auto (default: fp16 on GPU, fp32 on CPU), or one of "
             "float32/float16/bfloat16.",
    )


def add_output_arg(parser: argparse.ArgumentParser, default_name: str) -> None:
    """
    Add `--out`, so one run does not overwrite another's results.

    Every harness wrote to a single hardcoded path under benchmarks/results/,
    which means comparing two models required an external wrapper to copy the
    file away between runs -- and if a run failed after the previous one
    succeeded, the stale file was still sitting there looking current. Scaling
    to a second model made this the first thing that had to be worked around.
    """
    parser.add_argument(
        "--out", default=None, metavar="PATH",
        help=f"write the JSON here instead of "
             f"benchmarks/results/{default_name}.json",
    )


def resolve_device(args: argparse.Namespace) -> Tuple[str, torch.dtype]:
    """
    Turn `--device`/`--dtype` into a concrete pair, or fail loudly.

    Raises `SystemExit` with an explanation rather than returning something
    plausible, because a benchmark that quietly measures the wrong hardware is
    worse than one that refuses to start.
    """
    requested = str(args.device).lower()

    if requested == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit(
                f"--device {args.device} was requested but CUDA is not "
                f"available to this interpreter.\n"
                f"  torch        : {torch.__version__}\n"
                f"  torch.version.cuda: {torch.version.cuda}\n"
                f"This is usually a CPU-only torch wheel. Reinstall against a "
                f"CUDA build, e.g.\n"
                f"  pip install --force-reinstall torch "
                f"--index-url https://download.pytorch.org/whl/cu128\n"
                f"Refusing to fall back to CPU: the report would be labelled "
                f"cuda and contain CPU numbers."
            )
        index = 0
        if ":" in requested:
            try:
                index = int(requested.split(":", 1)[1])
            except ValueError:
                raise SystemExit(f"could not parse a device index from {args.device!r}")
        count = torch.cuda.device_count()
        if index >= count:
            raise SystemExit(
                f"--device {args.device} but this machine has {count} CUDA "
                f"device(s), so valid indices are 0..{count - 1}"
            )
        device = f"cuda:{index}"
    elif requested == "cpu":
        device = "cpu"
    else:
        raise SystemExit(f"unrecognised --device {args.device!r}; use auto, cuda or cpu")

    on_gpu = device.startswith("cuda")
    requested_dtype = str(args.dtype).lower()
    if requested_dtype == "auto":
        dtype = torch.float16 if on_gpu else torch.float32
    else:
        dtype = DTYPES[requested_dtype]

    if not on_gpu and dtype in (torch.float16, torch.bfloat16):
        # Not refused -- there are legitimate reasons to check numerics in
        # half precision on CPU -- but it must not be mistaken for a
        # performance measurement.
        print(
            f"  WARNING: {requested_dtype} on CPU has no hardware support. Any "
            f"timing below measures emulation, not the engine."
        )
    if on_gpu and dtype is torch.bfloat16 and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(index if on_gpu else 0)
        if major < 8:
            print(
                "  WARNING: bfloat16 needs compute capability 8.0+; this device "
                "will emulate it."
            )

    return device, dtype


def describe_device(device: str, dtype: torch.dtype) -> Dict[str, Any]:
    """
    The environment block for a benchmark's JSON report.

    Memory figures are `None` on CPU rather than 0, for the same reason the
    engine's telemetry reports them that way: torch has no CPU allocator to
    query, and a zero would read as "measured, and it was nothing" when nothing
    measured it.
    """
    payload: Dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
        "dtype": str(dtype),
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        index = int(device.split(":", 1)[1]) if ":" in device else 0
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        mb = 1024 * 1024
        payload.update(
            {
                "device_name": torch.cuda.get_device_name(index),
                "compute_capability": ".".join(
                    str(v) for v in torch.cuda.get_device_capability(index)
                ),
                "vram_total_mb": round(total_bytes / mb, 2),
                "vram_free_mb": round(free_bytes / mb, 2),
            }
        )
    else:
        payload.update(
            {
                "device_name": None,
                "compute_capability": None,
                "vram_total_mb": None,
                "vram_free_mb": None,
            }
        )
    return payload


def peak_memory_mb(device: str) -> Any:
    """
    Peak bytes the torch allocator held, in MiB, or None on CPU.

    None rather than 0.0 -- see `describe_device`. Phase E's definition of done
    asks for "real max_memory_allocated telemetry instead of the current zero
    on CPU", and reporting zero was precisely the misleading part.
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        index = int(device.split(":", 1)[1]) if ":" in device else 0
        return round(torch.cuda.max_memory_allocated(index) / (1024 * 1024), 2)
    return None


def vram_spill_check(device: str, peak_mb: Any) -> Dict[str, Any]:
    """
    Did this run exceed the card and quietly spill into system memory?

    On Windows, WDDM lets CUDA page device memory out to host RAM instead of
    failing an allocation. That is the worst available outcome for a benchmark:
    nothing raises, the JSON still says `cuda`, and the throughput collapses by
    3-5x because every access crosses PCIe. A context sweep here peaked at
    6687 MB on a 6141 MiB card and reported 3.73 tok/s as a GPU number -- the
    same run with a KV budget that fit measured 11.42.

    This is the same failure class `resolve_device` refuses for `--device cuda`
    on a CPU-only build: a plausible number carrying a wrong label. So it is
    reported in the payload and printed loudly rather than left for a reader to
    notice.
    """
    result: Dict[str, Any] = {"checked": False, "spilled": None, "peak_mb": peak_mb}
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        return result
    if not isinstance(peak_mb, (int, float)):
        return result
    index = int(device.split(":", 1)[1]) if ":" in device else 0
    total_mb = torch.cuda.get_device_properties(index).total_memory / (1024 * 1024)
    # 0.95 rather than 1.0: the driver and the display reserve a slice, so a
    # peak that close to the ceiling has already started evicting.
    spilled = peak_mb >= total_mb * 0.95
    result.update({
        "checked": True,
        "spilled": bool(spilled),
        "vram_total_mb": round(total_mb, 2),
        "threshold_mb": round(total_mb * 0.95, 2),
    })
    if spilled:
        print(
            f"\n  !! VRAM SPILL: peak allocation {peak_mb:.0f} MB against a "
            f"{total_mb:.0f} MB card.\n"
            f"     CUDA on Windows pages to host RAM instead of failing, so the "
            f"timings above are\n"
            f"     contaminated by PCIe traffic and are NOT valid GPU numbers. "
            f"Re-run with a smaller\n"
            f"     --kv-budget-mb, fewer --contexts, or --skip-batched."
        )
    return result


def reset_peak_memory(device: str) -> None:
    """Zero the high-water mark so one scenario's peak is not another's."""
    if device.startswith("cuda") and torch.cuda.is_available():
        index = int(device.split(":", 1)[1]) if ":" in device else 0
        torch.cuda.reset_peak_memory_stats(index)


def synchronize(device: str) -> None:
    """
    Wait for queued GPU work before stopping a timer.

    CUDA kernel launches are asynchronous, so a timer stopped without this
    measures how long it took to *queue* the work, not to do it -- which makes
    a GPU look absurdly fast and is the classic way to publish a wrong speedup.
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
