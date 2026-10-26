#!/usr/bin/env python3
"""
Multi-client traffic simulator.

Spawns N virtual users that behave roughly like real ones: random prompts,
random lengths, random think-time between requests, and -- the part that matters
-- an optional fraction of them hang up mid-stream.

How this differs from `benchmarks/run_stress_test.py`, since the two look
similar: that one is a **pass/fail gate** with fixed assertions and a stable
workload, suitable for CI. This one is an **exploratory generator** with random
behaviour and no assertions, for finding things the gate was not written to look
for. Both are worth having; neither replaces the other.

`--disconnect-rate` is the reason this script exists. Nothing else in the
project exercises client hangups under concurrent load, and that path -- HTTP
disconnect propagating into a scheduler abort so the physical blocks come back
-- is the newest and least-exercised part of the server. If block counts do not
return to baseline after a run with disconnects, that is a real leak and worth
reporting.

Usage:
    python scripts/traffic_simulator.py --concurrency 10 --requests 30
    python scripts/traffic_simulator.py --concurrency 20 --requests 100 --disconnect-rate 0.1
    python scripts/traffic_simulator.py --ws --concurrency 5 --requests 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
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
    import websockets
except ImportError:  # pragma: no cover
    websockets = None

RESULTS_PATH = PROJECT_ROOT / "benchmarks" / "results" / "traffic_simulation.json"

# A mix of lengths, and some sharing a long prefix so prefix sharing gets
# exercised rather than only the easy all-distinct case.
SHARED_PREAMBLE = (
    "You are MemForge, a careful and precise assistant. Answer accurately, "
    "prefer concrete detail over generalities, and say plainly when you do not "
    "know something. "
)

SHORT_PROMPTS = [
    "What is a KV cache?",
    "Define time to first token.",
    "What does a block table do?",
    "Why page the KV cache?",
    "What is continuous batching?",
]

LONG_PROMPTS = [
    SHARED_PREAMBLE + "Explain paged attention and why it reduces fragmentation.",
    SHARED_PREAMBLE + "Describe copy-on-write prefix sharing in an inference engine.",
    SHARED_PREAMBLE + "Why does speculative decoding need a cheap draft model?",
    SHARED_PREAMBLE + "How does preempt-by-recompute keep output identical?",
]


@dataclass
class Outcome:
    user: int
    index: int
    status: int = 0
    ttft_ms: Optional[float] = None
    total_s: float = 0.0
    tokens: int = 0
    disconnected: bool = False
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None and not self.disconnected


@dataclass
class Stats:
    outcomes: List[Outcome] = field(default_factory=list)

    def add(self, outcome: Outcome) -> None:
        self.outcomes.append(outcome)


# ---------------------------------------------------------------- transports

async def sse_request(client: httpx.AsyncClient, base_url: str, user: int, index: int,
                      prompt: str, tokens: int, disconnect_after: Optional[int]) -> Outcome:
    """
    One streaming request, optionally abandoned part-way through.

    Abandoning is done by breaking out of the response iterator, which closes
    the connection. The server should notice, abort the sequence and reclaim its
    blocks -- that is exactly the path being tested.
    """
    outcome = Outcome(user=user, index=index)
    started = time.perf_counter()
    first_at: Optional[float] = None

    try:
        async with client.stream(
            "POST", f"{base_url}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": prompt}],
                  "max_tokens": tokens, "stream": True},
            timeout=300,
        ) as response:
            outcome.status = response.status_code
            if response.status_code != 200:
                await response.aread()
                outcome.total_s = time.perf_counter() - started
                return outcome

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
                if "error" in chunk:
                    outcome.error = str(chunk["error"])[:200]
                    break
                delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                if delta.get("content"):
                    if first_at is None:
                        first_at = time.perf_counter()
                    outcome.tokens += 1
                    if disconnect_after is not None and outcome.tokens >= disconnect_after:
                        outcome.disconnected = True
                        break
    except Exception as exc:
        outcome.error = f"{type(exc).__name__}: {exc}"

    outcome.total_s = time.perf_counter() - started
    if first_at is not None:
        outcome.ttft_ms = (first_at - started) * 1000
    return outcome


async def ws_request(ws_url: str, user: int, index: int, prompt: str, tokens: int,
                     disconnect_after: Optional[int]) -> Outcome:
    outcome = Outcome(user=user, index=index)
    if websockets is None:
        outcome.error = "the websockets package is not installed"
        return outcome

    started = time.perf_counter()
    first_at: Optional[float] = None
    try:
        async with websockets.connect(ws_url, open_timeout=30) as socket:
            await socket.send(json.dumps({
                "prompt": prompt, "max_new_tokens": tokens,
                "temperature": 0.7, "top_p": 0.9,
            }))
            outcome.status = 200
            while True:
                event = json.loads(await asyncio.wait_for(socket.recv(), timeout=300))
                kind = event.get("type")
                if kind == "token":
                    if first_at is None:
                        first_at = time.perf_counter()
                    outcome.tokens += 1
                    if disconnect_after is not None and outcome.tokens >= disconnect_after:
                        outcome.disconnected = True
                        break
                elif kind == "done":
                    break
                elif kind == "error":
                    outcome.error = str(event.get("message"))[:200]
                    break
    except Exception as exc:
        outcome.error = f"{type(exc).__name__}: {exc}"

    outcome.total_s = time.perf_counter() - started
    if first_at is not None:
        outcome.ttft_ms = (first_at - started) * 1000
    return outcome


# ----------------------------------------------------------------- the users

async def virtual_user(user_id: int, args: argparse.Namespace, base_url: str,
                       ws_url: str, client: httpx.AsyncClient, stats: Stats,
                       rng: random.Random, budget: asyncio.Semaphore) -> None:
    for index in range(args.requests_per_user):
        # Think-time: real users do not fire requests back to back, and the gaps
        # are what let the scheduler drain between bursts.
        if index:
            await asyncio.sleep(rng.uniform(args.min_think, args.max_think))

        prompt = rng.choice(LONG_PROMPTS if rng.random() < args.long_prompt_rate
                            else SHORT_PROMPTS)
        tokens = rng.randint(args.min_tokens, args.max_tokens)

        disconnect_after: Optional[int] = None
        if rng.random() < args.disconnect_rate:
            # Hang up somewhere in the first part of the stream, which is the
            # interesting window: blocks are allocated and generation is live.
            disconnect_after = max(1, rng.randint(1, max(2, tokens // 3)))

        async with budget:
            if args.ws:
                outcome = await ws_request(ws_url, user_id, index, prompt, tokens,
                                           disconnect_after)
            else:
                outcome = await sse_request(client, base_url, user_id, index, prompt,
                                            tokens, disconnect_after)
        stats.add(outcome)

        if args.verbose:
            tag = ("DISCONNECT" if outcome.disconnected
                   else "ok" if outcome.ok else f"status {outcome.status}")
            print(f"    user {user_id:2} req {index:2}  {tag:11} "
                  f"{outcome.tokens:3} tok  {outcome.total_s:6.2f}s", flush=True)


# ---------------------------------------------------------------- reporting

def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


async def read_blocks(client: httpx.AsyncClient, base_url: str) -> Dict[str, int]:
    try:
        text = (await client.get(f"{base_url}/metrics", timeout=10)).text
    except Exception:
        return {}
    out: Dict[str, int] = {}
    for line in text.splitlines():
        if line.startswith("memforge_blocks_"):
            name, _, value = line.partition(" ")
            try:
                out[name] = int(float(value))
            except ValueError:
                continue
    return out


def report(stats: Stats, before: Dict[str, int], after: Dict[str, int],
           duration: float, args: argparse.Namespace) -> Dict[str, Any]:
    outcomes = stats.outcomes
    ok = [o for o in outcomes if o.ok]
    disconnected = [o for o in outcomes if o.disconnected]
    errored = [o for o in outcomes if o.error and not o.disconnected]

    status_hist: Dict[str, int] = {}
    for o in outcomes:
        key = "disconnected" if o.disconnected else (str(o.status) if o.status else "error")
        status_hist[key] = status_hist.get(key, 0) + 1

    ttfts = [o.ttft_ms for o in outcomes if o.ttft_ms is not None]
    rates = [o.tokens / o.total_s for o in ok if o.total_s > 0 and o.tokens]

    allocated_before = before.get("memforge_blocks_allocated", 0)
    allocated_after = after.get("memforge_blocks_allocated", 0)
    leaked = after.get("memforge_blocks_leaked", 0)

    print()
    print("=" * 74)
    print("  Traffic simulation summary")
    print("=" * 74)

    def row(label: str, value: Any) -> None:
        print(f"  {label:34} {value}")

    row("virtual users", args.concurrency)
    row("requests sent", len(outcomes))
    row("completed normally", len(ok))
    row("disconnected deliberately", len(disconnected))
    row("errors", len(errored))
    row("status histogram", status_hist)
    row("duration", f"{duration:.1f} s")
    print()
    row("blocks allocated before", allocated_before)
    row("blocks allocated after", allocated_after)
    row("blocks leaked (must be 0)", leaked)
    print()
    row("TTFT p50", f"{percentile(ttfts, 50):.1f} ms" if ttfts else "--")
    row("TTFT p95", f"{percentile(ttfts, 95):.1f} ms" if ttfts else "--")
    row("TTFT p99", f"{percentile(ttfts, 99):.1f} ms" if ttfts else "--")
    row("mean tokens/sec per request",
        f"{statistics.mean(rates):.2f}" if rates else "--")
    row("total tokens streamed", sum(o.tokens for o in outcomes))

    print()
    print("=" * 74)
    verdicts: List[str] = []

    if leaked == 0:
        verdicts.append("PASS  no leaked blocks")
    else:
        verdicts.append(f"FAIL  {leaked} leaked blocks")

    if allocated_after <= allocated_before:
        verdicts.append("PASS  all physical blocks returned to the pool")
    else:
        verdicts.append(
            f"FAIL  {allocated_after - allocated_before} blocks still allocated. "
            "With disconnects in the run this points at the HTTP-disconnect abort path."
        )

    unexpected = {k: v for k, v in status_hist.items()
                  if k not in ("200", "429", "disconnected")}
    if unexpected:
        verdicts.append(f"WARN  unexpected outcomes {unexpected}")
    else:
        verdicts.append("PASS  only 200s, 429s and deliberate disconnects")

    if disconnected:
        verdicts.append(
            f"NOTE  {len(disconnected)} clients hung up mid-stream; the block counts "
            "above are the check that the server noticed"
        )

    for verdict in verdicts:
        print(f"  {verdict}")
    print("=" * 74)

    return {
        "simulator": "traffic",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "url": args.url,
            "transport": "websocket" if args.ws else "sse",
            "concurrency": args.concurrency,
            "requests_per_user": args.requests_per_user,
            "total_requests": len(outcomes),
            "disconnect_rate": args.disconnect_rate,
            "tokens": [args.min_tokens, args.max_tokens],
            "think_s": [args.min_think, args.max_think],
            "seed": args.seed,
        },
        "duration_s": round(duration, 2),
        "requests_sent": len(outcomes),
        "completed": len(ok),
        "disconnected": len(disconnected),
        "errors": len(errored),
        "status_histogram": status_hist,
        "blocks_allocated_before": allocated_before,
        "blocks_allocated_after": allocated_after,
        "leaked_blocks": leaked,
        "p50_ttft_ms": round(percentile(ttfts, 50), 2) if ttfts else None,
        "p95_ttft_ms": round(percentile(ttfts, 95), 2) if ttfts else None,
        "p99_ttft_ms": round(percentile(ttfts, 99), 2) if ttfts else None,
        "mean_tok_per_sec": round(statistics.mean(rates), 2) if rates else None,
        "total_tokens": sum(o.tokens for o in outcomes),
        "verdicts": verdicts,
        "outcomes": [
            {"user": o.user, "index": o.index, "status": o.status,
             "ttft_ms": round(o.ttft_ms, 2) if o.ttft_ms else None,
             "total_s": round(o.total_s, 3), "tokens": o.tokens,
             "disconnected": o.disconnected, "error": o.error}
            for o in outcomes
        ],
    }


# --------------------------------------------------------------------- main

async def run(args: argparse.Namespace) -> int:
    base_url = args.url.rstrip("/")
    ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://") + "/ws/stream"

    async with httpx.AsyncClient() as client:
        try:
            health = (await client.get(f"{base_url}/v1/health", timeout=15)).json()
        except Exception as exc:
            print(f"  cannot reach {base_url}: {type(exc).__name__}: {exc}", file=sys.stderr)
            print("  start one with: uvicorn server.app:app --port 8000", file=sys.stderr)
            return 2
        if not health.get("engine_ready"):
            print(f"  engine not ready (load_error={health.get('load_error')})",
                  file=sys.stderr)
            return 2

        before = await read_blocks(client, base_url)

        print(f"  {args.concurrency} virtual users x {args.requests_per_user} requests "
              f"over {'WebSocket' if args.ws else 'SSE'}")
        if args.disconnect_rate:
            print(f"  {args.disconnect_rate:.0%} of requests will hang up mid-stream")

        # In-flight cap, separate from the user count: users also spend time
        # thinking, so the two are not the same number.
        budget = asyncio.Semaphore(args.max_in_flight or args.concurrency)
        stats = Stats()

        started = time.perf_counter()
        await asyncio.gather(*(
            virtual_user(user_id, args, base_url, ws_url, client, stats,
                         random.Random(args.seed + user_id), budget)
            for user_id in range(args.concurrency)
        ))
        duration = time.perf_counter() - started

        # Let the scheduler reap aborted sequences before the final reading.
        await asyncio.sleep(2.0)
        after = await read_blocks(client, base_url)

    payload = report(stats, before, after, duration, args)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  wrote {RESULTS_PATH.relative_to(PROJECT_ROOT)}")

    failed = any(v.startswith("FAIL") for v in payload["verdicts"])
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="MemForge multi-client traffic simulator")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=10, help="virtual users")
    parser.add_argument("--requests", type=int, default=30,
                        help="total requests, split across users")
    parser.add_argument("--max-in-flight", type=int, default=0,
                        help="cap on simultaneous requests (default: the user count)")
    parser.add_argument("--disconnect-rate", type=float, default=0.0,
                        help="fraction of requests that hang up mid-stream (0..1)")
    parser.add_argument("--min-tokens", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--min-think", type=float, default=0.0)
    parser.add_argument("--max-think", type=float, default=0.4)
    parser.add_argument("--long-prompt-rate", type=float, default=0.5,
                        help="fraction using the shared long preamble, exercising prefix sharing")
    parser.add_argument("--ws", action="store_true", help="use /ws/stream instead of SSE")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true", help="log every request")
    args = parser.parse_args()

    if not 0.0 <= args.disconnect_rate <= 1.0:
        parser.error("--disconnect-rate must be between 0 and 1")
    args.concurrency = max(1, args.concurrency)
    args.requests_per_user = max(1, args.requests // args.concurrency)

    print("=" * 74)
    print("  MemForge traffic simulator")
    print("=" * 74)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
