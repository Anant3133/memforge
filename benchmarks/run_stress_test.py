#!/usr/bin/env python3
"""
Sustained load and memory-leak harness.

This is the measurement Phase F's definition of done depends on: *"the server
survives a sustained multi-client load test with flat memory and no leaked
blocks."* Until this runs, that claim cannot be made.

Three measurements, in descending order of how much they can be trusted:

  1. **Block accounting** -- exact. `/metrics` exposes `memforge_blocks_leaked`
     as `total - free - allocated`, which must be 0 at every sample, and
     `memforge_blocks_allocated` must return to its starting value once the run
     drains. There is no warm-up behaviour to explain away, so a failure here is
     a real leak.
  2. **Latency percentiles** -- reliable in aggregate. TTFT is measured
     client-side, which is the latency a caller actually experiences.
  3. **Process RSS** -- indicative only. PyTorch and the allocator both cache,
     so RSS rises over the first requests and then plateaus. That is normal, and
     this harness compares the final value against a plateau measured *after*
     warm-up rather than against the first sample, because comparing against the
     first sample reports a leak on every healthy run.

Usage:
    # Simplest: let the harness own the server lifecycle.
    python benchmarks/run_stress_test.py --spawn-server --requests 50 --concurrency 8

    # Against a server you started yourself.
    python benchmarks/run_stress_test.py --url http://127.0.0.1:8000 --requests 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx  # noqa: E402

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

RESULTS_PATH = PROJECT_ROOT / "benchmarks" / "results" / "stress_test.json"

PROMPTS = [
    "Explain paged attention in two sentences.",
    "What is continuous batching and why does it help?",
    "Summarise copy-on-write prefix sharing.",
    "Why does speculative decoding need a cheap draft model?",
    "Describe a KV cache block table.",
    "What is time to first token?",
    "How does preempt-by-recompute work?",
    "Why is fragmentation less of a problem with paging?",
]


# --------------------------------------------------------------- data classes

@dataclass
class RequestResult:
    index: int
    status: int
    ttft_ms: Optional[float] = None
    total_s: float = 0.0
    tokens: int = 0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None


@dataclass
class Sample:
    at: float
    rss_mb: Optional[float]
    blocks_total: int
    blocks_allocated: int
    blocks_free: int
    blocks_leaked: int
    blocks_shared: int
    in_flight: int


@dataclass
class Harness:
    base_url: str
    results: List[RequestResult] = field(default_factory=list)
    samples: List[Sample] = field(default_factory=list)
    process: Optional[Any] = None  # psutil.Process for the server

    # ------------------------------------------------------------- sampling

    def read_rss_mb(self) -> Optional[float]:
        """
        Resident memory of the whole server process tree.

        Summing the tree rather than reading one process is not a detail. On
        Windows, `python -m uvicorn` is a ~5 MB launcher that spawns a child
        holding the actual model: measured here, the parent sat at 5.6 MB while
        its child held 1002.7 MB. Sampling only the parent therefore reports a
        perfectly flat 5.6 MB no matter what the server does, which passes the
        leak check for entirely the wrong reason. A memory harness that cannot
        see the memory is worse than none, because it produces false confidence.
        """
        if self.process is None:
            return None
        try:
            total = self.process.memory_info().rss
            for child in self.process.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except Exception:
                    # A worker that exited between listing and reading is fine
                    # to skip; it is no longer holding anything.
                    continue
            return total / 1024**2
        except Exception:
            return None

    async def scrape(self, client: httpx.AsyncClient) -> Optional[Sample]:
        """One point-in-time reading of the server's memory accounting."""
        try:
            text = (await client.get(f"{self.base_url}/metrics", timeout=10)).text
        except Exception:
            return None

        gauges: Dict[str, float] = {}
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            name, _, value = line.partition(" ")
            try:
                gauges[name] = float(value)
            except ValueError:
                continue

        rss = self.read_rss_mb()

        sample = Sample(
            at=time.perf_counter(),
            rss_mb=rss,
            blocks_total=int(gauges.get("memforge_blocks_total", 0)),
            blocks_allocated=int(gauges.get("memforge_blocks_allocated", 0)),
            blocks_free=int(gauges.get("memforge_blocks_free", 0)),
            blocks_leaked=int(gauges.get("memforge_blocks_leaked", 0)),
            blocks_shared=int(gauges.get("memforge_blocks_shared", 0)),
            in_flight=int(gauges.get("memforge_requests_in_flight", 0)),
        )
        self.samples.append(sample)
        return sample

    async def sampler_loop(self, client: httpx.AsyncClient, stop: asyncio.Event,
                           interval: float) -> None:
        while not stop.is_set():
            await self.scrape(client)
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    # ------------------------------------------------------------- requests

    async def one_request(self, client: httpx.AsyncClient, index: int,
                          tokens: int, stream: bool) -> RequestResult:
        prompt = random.choice(PROMPTS)
        started = time.perf_counter()

        if not stream:
            try:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": prompt}],
                          "max_tokens": tokens},
                    timeout=300,
                )
            except Exception as exc:
                return RequestResult(index, 0, total_s=time.perf_counter() - started,
                                     error=f"{type(exc).__name__}: {exc}")

            elapsed = time.perf_counter() - started
            if response.status_code != 200:
                return RequestResult(index, response.status_code, total_s=elapsed)
            body = response.json()
            # Non-streaming has no observable first token, so the server's own
            # figure is the only one available; it is flagged as such.
            server_ttft = (body.get("memforge_telemetry") or {}).get("time_to_first_token_ms")
            return RequestResult(index, 200, ttft_ms=server_ttft, total_s=elapsed,
                                 tokens=body.get("usage", {}).get("completion_tokens", 0))

        # Streaming: TTFT measured here, which is what a caller experiences.
        first_at: Optional[float] = None
        count = 0
        try:
            async with client.stream(
                "POST", f"{self.base_url}/v1/chat/completions",
                json={"messages": [{"role": "user", "content": prompt}],
                      "max_tokens": tokens, "stream": True},
                timeout=300,
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    return RequestResult(index, response.status_code,
                                         total_s=time.perf_counter() - started)
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                    if delta.get("content"):
                        if first_at is None:
                            first_at = time.perf_counter()
                        count += 1
        except Exception as exc:
            return RequestResult(index, 0, total_s=time.perf_counter() - started,
                                 error=f"{type(exc).__name__}: {exc}")

        return RequestResult(
            index, 200,
            ttft_ms=(first_at - started) * 1000 if first_at else None,
            total_s=time.perf_counter() - started,
            tokens=count,
        )

    async def run_phase(self, client: httpx.AsyncClient, count: int, concurrency: int,
                        tokens: int, stream: bool, label: str) -> None:
        print(f"  {label}: {count} requests at concurrency {concurrency}…", flush=True)
        semaphore = asyncio.Semaphore(concurrency)
        base = len(self.results)

        async def guarded(i: int) -> RequestResult:
            async with semaphore:
                return await self.one_request(client, base + i, tokens, stream)

        results = await asyncio.gather(*(guarded(i) for i in range(count)))
        self.results.extend(results)


# ---------------------------------------------------------------- reporting

def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100) * (len(ordered) - 1)
    low, high = int(rank), min(int(rank) + 1, len(ordered) - 1)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def summarise(harness: Harness, warmup: int) -> Dict[str, Any]:
    results = harness.results
    samples = harness.samples
    ok = [r for r in results if r.ok]
    ttfts = [r.ttft_ms for r in ok if r.ttft_ms is not None]

    status_counts: Dict[str, int] = {}
    for r in results:
        key = str(r.status) if r.status else f"error:{(r.error or '').split(':')[0]}"
        status_counts[key] = status_counts.get(key, 0) + 1

    rss_values = [s.rss_mb for s in samples if s.rss_mb is not None]
    # Compare the tail against a post-warm-up plateau, not against the first
    # sample: PyTorch and the allocator cache, so a healthy run always grows at
    # the start and comparing to sample 0 reports a leak every time.
    plateau_from = min(warmup, max(0, len(rss_values) - 1))
    plateau = rss_values[plateau_from:] if rss_values else []

    leaked_samples = [s.blocks_leaked for s in samples]
    allocated_first = samples[0].blocks_allocated if samples else 0
    allocated_last = samples[-1].blocks_allocated if samples else 0

    tok_rates = [r.tokens / r.total_s for r in ok if r.total_s > 0 and r.tokens]

    return {
        "total_requests": len(results),
        "succeeded": len(ok),
        "failed": len(results) - len(ok),
        "status_counts": status_counts,
        "initial_rss_mb": round(rss_values[0], 1) if rss_values else None,
        "plateau_rss_mb": round(statistics.median(plateau), 1) if plateau else None,
        "peak_rss_mb": round(max(rss_values), 1) if rss_values else None,
        "final_rss_mb": round(rss_values[-1], 1) if rss_values else None,
        "rss_growth_after_warmup_mb": (
            round(rss_values[-1] - statistics.median(plateau), 1) if plateau else None
        ),
        "leaked_blocks": max(leaked_samples) if leaked_samples else 0,
        "blocks_allocated_at_start": allocated_first,
        "blocks_allocated_at_end": allocated_last,
        "peak_blocks_allocated": max((s.blocks_allocated for s in samples), default=0),
        "peak_blocks_shared": max((s.blocks_shared for s in samples), default=0),
        "peak_in_flight": max((s.in_flight for s in samples), default=0),
        "p50_ttft_ms": round(percentile(ttfts, 50), 2) if ttfts else None,
        "p95_ttft_ms": round(percentile(ttfts, 95), 2) if ttfts else None,
        "p99_ttft_ms": round(percentile(ttfts, 99), 2) if ttfts else None,
        "mean_tok_per_sec": round(statistics.mean(tok_rates), 2) if tok_rates else None,
        "total_tokens": sum(r.tokens for r in ok),
        "samples_taken": len(samples),
    }


def print_report(summary: Dict[str, Any], duration: float, verdicts: List[str]) -> None:
    print()
    print("=" * 74)
    print("  Stress & leak test summary")
    print("=" * 74)

    def row(label: str, value: Any) -> None:
        print(f"  {label:34} {value}")

    row("requests", f"{summary['succeeded']} ok / {summary['total_requests']} sent")
    row("status codes", summary["status_counts"])
    row("duration", f"{duration:.1f} s")
    print()
    row("blocks leaked (must be 0)", summary["leaked_blocks"])
    row("blocks allocated start -> end",
        f"{summary['blocks_allocated_at_start']} -> {summary['blocks_allocated_at_end']}")
    row("peak blocks allocated", summary["peak_blocks_allocated"])
    row("peak blocks shared", summary["peak_blocks_shared"])
    row("peak requests in flight", summary["peak_in_flight"])
    print()
    row("RSS initial", f"{summary['initial_rss_mb']} MB")
    row("RSS plateau (post-warm-up)", f"{summary['plateau_rss_mb']} MB")
    row("RSS peak", f"{summary['peak_rss_mb']} MB")
    row("RSS final", f"{summary['final_rss_mb']} MB")
    row("RSS growth vs plateau", f"{summary['rss_growth_after_warmup_mb']} MB")
    print()
    row("TTFT p50 / p95 / p99",
        f"{summary['p50_ttft_ms']} / {summary['p95_ttft_ms']} / {summary['p99_ttft_ms']} ms")
    row("mean tokens/sec per request", summary["mean_tok_per_sec"])
    row("total tokens generated", summary["total_tokens"])

    print()
    print("=" * 74)
    for verdict in verdicts:
        print(f"  {verdict}")
    print("=" * 74)


# ------------------------------------------------------------ server control

def spawn_server(port: int, kv_budget_mb: float, max_concurrency: int,
                 model: Optional[str]) -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("MEMFORGE_LOAD_DRAFT", "false")
    env["MEMFORGE_KV_BUDGET_MB"] = str(kv_budget_mb)
    env["MEMFORGE_MAX_CONCURRENCY"] = str(max_concurrency)
    if model:
        env["MEMFORGE_TARGET_MODEL"] = model

    print(f"  spawning a server on port {port} (model "
          f"{model or env.get('MEMFORGE_TARGET_MODEL', 'default')})…", flush=True)
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=str(PROJECT_ROOT), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


async def wait_ready(base_url: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(timeout=10) as client:
        while time.monotonic() < deadline:
            try:
                health = (await client.get(f"{base_url}/v1/health")).json()
                if health.get("load_error"):
                    print(f"  engine failed to load: {health['load_error']}")
                    return False
                if health.get("engine_ready"):
                    return True
            except Exception:
                pass
            await asyncio.sleep(1.0)
    return False


# --------------------------------------------------------------------- main

async def run(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    child: Optional[subprocess.Popen] = None
    base_url = args.url.rstrip("/")

    if args.spawn_server:
        child = spawn_server(args.port, args.kv_budget_mb, args.max_concurrency, args.model)
        base_url = f"http://127.0.0.1:{args.port}"

    try:
        print(f"  waiting for {base_url} to become ready…", flush=True)
        if not await wait_ready(base_url, args.startup_timeout):
            print("  server never became ready", file=sys.stderr)
            return 2

        harness = Harness(base_url=base_url)

        # RSS needs the server's pid. When we spawned it we know it; otherwise
        # find the listener on the port. If neither works, block accounting
        # still carries the leak verdict on its own.
        if psutil is not None:
            try:
                if child is not None:
                    harness.process = psutil.Process(child.pid)
                else:
                    port = int(base_url.rsplit(":", 1)[-1])
                    for conn in psutil.net_connections(kind="inet"):
                        if conn.laddr and conn.laddr.port == port and conn.pid:
                            harness.process = psutil.Process(conn.pid)
                            break
            except Exception:
                harness.process = None
        if harness.process is None:
            print("  note: server pid not resolved, so RSS is not sampled; "
                  "block accounting still applies", flush=True)

        async with httpx.AsyncClient() as client:
            stop = asyncio.Event()
            sampler = asyncio.create_task(
                harness.sampler_loop(client, stop, args.sample_interval)
            )
            await harness.scrape(client)  # baseline before any load

            started = time.perf_counter()
            await harness.run_phase(client, args.warmup, 1, args.tokens,
                                    args.stream, "warm-up")
            warmup_samples = len(harness.samples)

            await harness.run_phase(client, args.requests // 2, 1, args.tokens,
                                    args.stream, "sequential")
            await harness.run_phase(client, args.requests - args.requests // 2,
                                    args.concurrency, args.tokens, args.stream,
                                    "concurrent")
            duration = time.perf_counter() - started

            # Let the engine drain so the final block reading is post-teardown.
            await asyncio.sleep(2.0)
            await harness.scrape(client)
            stop.set()
            await sampler

        summary = summarise(harness, warmup_samples)

        verdicts: List[str] = []
        failed = False

        if summary["leaked_blocks"] == 0:
            verdicts.append("PASS  no leaked blocks at any sample")
        else:
            verdicts.append(f"FAIL  {summary['leaked_blocks']} leaked blocks observed")
            failed = True

        if summary["blocks_allocated_at_end"] <= summary["blocks_allocated_at_start"]:
            verdicts.append("PASS  all physical blocks returned to the pool")
        else:
            verdicts.append(
                f"FAIL  {summary['blocks_allocated_at_end'] - summary['blocks_allocated_at_start']}"
                " blocks still allocated after the run drained"
            )
            failed = True

        if summary["succeeded"] == 0:
            verdicts.append("FAIL  no request succeeded")
            failed = True
        elif summary["failed"]:
            unexpected = {k: v for k, v in summary["status_counts"].items()
                          if k not in ("200", "429")}
            if unexpected:
                verdicts.append(f"FAIL  unexpected statuses {unexpected}")
                failed = True
            else:
                verdicts.append("PASS  only 200s and 429s (429 is backpressure, not failure)")

        growth = summary["rss_growth_after_warmup_mb"]
        if growth is None:
            verdicts.append("SKIP  RSS not sampled (server pid unavailable)")
        elif growth <= args.rss_tolerance_mb:
            verdicts.append(f"PASS  RSS flat after warm-up ({growth:+.1f} MB)")
        else:
            # Deliberately not a failure: RSS is indicative, and block
            # accounting is the exact measurement. Flagged for a human.
            verdicts.append(
                f"WARN  RSS grew {growth:+.1f} MB after warm-up "
                f"(tolerance {args.rss_tolerance_mb} MB) - check against the block counts"
            )

        print_report(summary, duration, verdicts)

        payload = {
            "benchmark": "stress_test",
            "phase": "F",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "platform": {"system": platform.system(), "python": platform.python_version()},
            "config": {
                "url": base_url,
                "requests": args.requests,
                "concurrent_clients": args.concurrency,
                "warmup": args.warmup,
                "tokens": args.tokens,
                "stream": args.stream,
                "kv_budget_mb": args.kv_budget_mb if args.spawn_server else None,
                "max_concurrency": args.max_concurrency if args.spawn_server else None,
            },
            "duration_s": round(duration, 2),
            "passed": not failed,
            "verdicts": verdicts,
            **summary,
            "samples": [
                {
                    "t": round(s.at, 3), "rss_mb": s.rss_mb,
                    "blocks_allocated": s.blocks_allocated,
                    "blocks_free": s.blocks_free,
                    "blocks_leaked": s.blocks_leaked,
                    "blocks_shared": s.blocks_shared,
                    "in_flight": s.in_flight,
                }
                for s in harness.samples
            ],
        }
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n  wrote {RESULTS_PATH.relative_to(PROJECT_ROOT)}")

        return 1 if failed else 0

    finally:
        if child is not None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="MemForge sustained load & leak harness")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5,
                        help="requests run before RSS plateau measurement begins")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--stream", action="store_true",
                        help="use SSE so TTFT is measured client-side")
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--rss-tolerance-mb", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spawn-server", action="store_true",
                        help="start and stop a server for this run")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--kv-budget-mb", type=float, default=192.0)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--model", default=None, help="MEMFORGE_TARGET_MODEL for --spawn-server")
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    args = parser.parse_args()

    print("=" * 74)
    print("  MemForge sustained stress & memory-leak harness")
    print("=" * 74)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
