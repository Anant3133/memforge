#!/usr/bin/env python3
"""
Interactive terminal chat client for a running MemForge server.

Exists so that testing a change does not mean opening a browser or hand-writing
a curl invocation with a JSON body. Streams tokens as they arrive, colours them
by provenance, and prints the three numbers that actually matter after each
answer: time to first token, tokens per second, and total wall time.

Transport is the WebSocket at /ws/stream by default, because it carries each
token's `source` field and so makes speculative decoding visible; `--sse` uses
the OpenAI-compatible Server-Sent Events path instead, which is the one external
clients would use.

Timings are measured here, on the client, not read from the server's `done`
event. The server's figure excludes transport, and TTFT as experienced is what
matters when you are deciding whether a change made things better.

Usage:
    python scripts/memforge_cli.py
    python scripts/memforge_cli.py --url http://127.0.0.1:8111 --tokens 64
    python scripts/memforge_cli.py --sse --spec
    python scripts/memforge_cli.py --prompt "one shot question" --tokens 32
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import urlparse

try:
    import httpx
except ImportError:  # pragma: no cover
    print("httpx is required: pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(1)

try:
    import websockets
except ImportError:  # pragma: no cover
    websockets = None

try:
    from rich.console import Console
    from rich.panel import Panel

    CONSOLE: Optional[Console] = Console()
except ImportError:  # pragma: no cover
    CONSOLE = None


SOURCE_STYLE = {
    "target": "white",
    "draft": "bright_cyan",
    "resample": "bright_yellow",
    "bonus": "bright_magenta",
}

HELP = """
  /help            show this
  /spec            toggle speculative decoding
  /tokens N        set max new tokens
  /temp X          set temperature
  /stats           print engine telemetry
  /blocks          print the physical block map
  /metrics         print the Prometheus gauges
  /reset           clear the conversation history
  /quit            exit
"""


def emit(text: str = "", style: Optional[str] = None, end: str = "\n") -> None:
    """Print through rich when available, falling back to plain stdout."""
    if CONSOLE is not None:
        CONSOLE.print(text, style=style, end=end, highlight=False, soft_wrap=True)
    else:
        sys.stdout.write(text + end)
        sys.stdout.flush()


def emit_raw(text: str, style: Optional[str] = None) -> None:
    """Write a token without a newline, keeping the stream flowing."""
    if CONSOLE is not None:
        CONSOLE.print(text, style=style, end="", highlight=False, soft_wrap=True)
    else:
        sys.stdout.write(text)
        sys.stdout.flush()


@dataclass
class RunStats:
    """Client-measured timings for one response."""

    started: float
    first_token_at: Optional[float] = None
    last_token_at: Optional[float] = None
    tokens: int = 0
    text: str = ""
    sources: List[str] = field(default_factory=list)
    error: Optional[str] = None
    finish_reason: Optional[str] = None

    def record(self, token: str, source: str = "target") -> None:
        now = time.perf_counter()
        if self.first_token_at is None:
            self.first_token_at = now
        self.last_token_at = now
        self.tokens += 1
        self.text += token
        self.sources.append(source)

    @property
    def ttft_ms(self) -> Optional[float]:
        if self.first_token_at is None:
            return None
        return (self.first_token_at - self.started) * 1000

    @property
    def total_s(self) -> float:
        end = self.last_token_at or time.perf_counter()
        return end - self.started

    @property
    def decode_tok_per_s(self) -> Optional[float]:
        """
        Steady-state decode rate, excluding the wait for the first token.

        Reported separately from the overall rate on purpose: including TTFT
        conflates prefill cost with decode throughput, and they move for
        different reasons.
        """
        if self.first_token_at is None or self.last_token_at is None or self.tokens < 2:
            return None
        span = self.last_token_at - self.first_token_at
        return (self.tokens - 1) / span if span > 0 else None

    @property
    def overall_tok_per_s(self) -> Optional[float]:
        return self.tokens / self.total_s if self.total_s > 0 and self.tokens else None


def print_stats(stats: RunStats, metrics: Optional[dict]) -> None:
    if stats.error:
        emit(f"\n  error: {stats.error}", style="bold red")
        return

    parts = []
    if stats.ttft_ms is not None:
        parts.append(f"TTFT {stats.ttft_ms:.0f} ms")
    parts.append(f"{stats.tokens} token{'' if stats.tokens == 1 else 's'}")
    if stats.decode_tok_per_s:
        parts.append(f"{stats.decode_tok_per_s:.1f} tok/s decode")
    if stats.overall_tok_per_s:
        parts.append(f"{stats.overall_tok_per_s:.1f} tok/s overall")
    parts.append(f"{stats.total_s:.2f} s total")
    if stats.finish_reason:
        parts.append(stats.finish_reason)

    emit("\n  " + "   |   ".join(parts), style="dim")

    drafted = sum(1 for s in stats.sources if s == "draft")
    if drafted:
        rate = drafted / len(stats.sources) * 100
        emit(f"  {drafted}/{len(stats.sources)} tokens from the draft model "
             f"({rate:.0f}% accepted)", style="dim cyan")

    if metrics:
        bits = [
            f"blocks {metrics.get('allocated_blocks', '?')}/{metrics.get('total_blocks', '?')}",
            f"shared {metrics.get('shared_blocks', 0)}",
            f"saved {metrics.get('memory_saved_mb', 0)} MB",
            f"tok/pass {metrics.get('tokens_per_forward_pass', 0)}",
        ]
        if metrics.get("in_flight") is not None:
            bits.append(f"in flight {metrics['in_flight']}/{metrics.get('max_concurrency', '?')}")
        emit("  " + "   |   ".join(bits), style="dim")


# --------------------------------------------------------------- transports

async def stream_over_websocket(ws_url: str, prompt: str, max_tokens: int,
                                temperature: float, use_spec: bool,
                                raw: bool = False) -> RunStats:
    if websockets is None:
        raise RuntimeError("the websockets package is required for --ws (the default)")

    stats = RunStats(started=time.perf_counter())
    metrics: Optional[dict] = None

    async with websockets.connect(ws_url, open_timeout=10) as socket:
        await socket.send(json.dumps({
            "prompt": prompt,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "use_speculative": use_spec,
            # The server applies the model's chat template unless told not to.
            # A multi-turn prompt is already role-marked here, so it must not be
            # wrapped a second time.
            "raw": raw,
        }))

        while True:
            raw = await asyncio.wait_for(socket.recv(), timeout=300)
            event = json.loads(raw)
            kind = event.get("type")

            if kind == "token":
                source = event.get("source", "target")
                stats.record(event.get("token", ""), source)
                emit_raw(event["token"], SOURCE_STYLE.get(source, "white"))
                metrics = event.get("metrics") or metrics
            elif kind == "done":
                stats.finish_reason = event.get("finish_reason")
                metrics = event.get("metrics") or metrics
                break
            elif kind == "error":
                stats.error = event.get("message", "unknown error")
                break
            # "start" needs no handling; it only echoes the request back.

    stats_metrics = metrics
    print_stats(stats, stats_metrics)
    return stats


async def stream_over_sse(base_url: str, messages: List[dict], max_tokens: int,
                          temperature: float, use_spec: bool) -> RunStats:
    """
    The OpenAI-compatible path, which is what an external client would use.

    Also the path that exercises disconnect handling: closing this stream early
    must make the server abort the sequence and reclaim its blocks.
    """
    stats = RunStats(started=time.perf_counter())
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "use_speculative": use_spec,
    }

    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream("POST", f"{base_url}/v1/chat/completions",
                                 json=payload) as response:
            if response.status_code == 429:
                await response.aread()
                retry = response.headers.get("Retry-After", "1")
                stats.error = f"engine at capacity (429); retry after {retry}s"
                print_stats(stats, None)
                return stats
            if response.status_code >= 400:
                body = await response.aread()
                stats.error = f"HTTP {response.status_code}: {body.decode(errors='replace')[:200]}"
                print_stats(stats, None)
                return stats

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if "error" in chunk:
                    stats.error = chunk["error"]
                    break

                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                token = delta.get("content")
                if token:
                    source = delta.get("memforge_source", "target")
                    stats.record(token, source)
                    emit_raw(token, SOURCE_STYLE.get(source, "white"))
                if choice.get("finish_reason"):
                    stats.finish_reason = choice["finish_reason"]

    print_stats(stats, None)
    return stats


# ------------------------------------------------------------ server probing

async def wait_for_engine(base_url: str, timeout: float = 120.0) -> bool:
    """
    Block until the engine reports ready.

    A cold server spends 10-30 s loading weights and answers /v1/health with
    engine_ready: false throughout. Polling here turns that into a progress
    message instead of a confusing failure on the first prompt.
    """
    deadline = time.monotonic() + timeout
    announced = False
    async with httpx.AsyncClient(timeout=10) as client:
        while time.monotonic() < deadline:
            try:
                health = (await client.get(f"{base_url}/v1/health")).json()
            except Exception as exc:
                emit(f"cannot reach {base_url} ({type(exc).__name__}).", style="bold red")
                emit("Is the server running?  uvicorn server.app:app --port 8000",
                     style="dim")
                return False

            if health.get("load_error"):
                emit(f"engine failed to load: {health['load_error']}", style="bold red")
                return False
            if health.get("engine_ready"):
                if announced:
                    emit("engine ready.", style="green")
                return True

            if not announced:
                emit("engine is loading model weights (10-30 s on a cold start)…",
                     style="yellow")
                announced = True
            await asyncio.sleep(1.0)

    emit("timed out waiting for the engine to become ready.", style="bold red")
    return False


async def show_json(base_url: str, path: str, title: str) -> None:
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            response = await client.get(f"{base_url}{path}")
            if response.status_code != 200:
                emit(f"{path} -> HTTP {response.status_code}", style="red")
                return
            body = json.dumps(response.json(), indent=2)
        except Exception as exc:
            emit(f"{path} failed: {exc}", style="red")
            return
    if CONSOLE is not None:
        CONSOLE.print(Panel(body, title=title, border_style="cyan"))
    else:
        emit(f"--- {title} ---\n{body}")


async def show_metrics(base_url: str) -> None:
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            text = (await client.get(f"{base_url}/metrics")).text
        except Exception as exc:
            emit(f"/metrics failed: {exc}", style="red")
            return
    gauges = "\n".join(l for l in text.splitlines() if l and not l.startswith("#"))
    if CONSOLE is not None:
        CONSOLE.print(Panel(gauges, title="/metrics", border_style="cyan"))
    else:
        emit(gauges)


# ------------------------------------------------------------------ the loop

async def repl(args: argparse.Namespace) -> int:
    base_url = args.url.rstrip("/")
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_url = f"{scheme}://{parsed.netloc}/ws/stream"

    if not await wait_for_engine(base_url):
        return 1

    max_tokens, temperature, use_spec = args.tokens, args.temperature, args.spec
    history: List[dict] = []

    if CONSOLE is not None:
        CONSOLE.print(Panel.fit(
            "[bold cyan]MemForge CLI[/bold cyan]\n"
            f"[dim]{base_url}  |  {'SSE' if args.sse else 'WebSocket'}  |  "
            f"/help for commands[/dim]",
            border_style="cyan"))
    else:
        emit(f"MemForge CLI — {base_url} — /help for commands")

    # One-shot mode: useful in scripts and for a quick smoke test.
    if args.prompt:
        emit("\nMemForge> ", style="bold green", end="")
        if args.sse:
            await stream_over_sse(base_url, [{"role": "user", "content": args.prompt}],
                                  max_tokens, temperature, use_spec)
        else:
            await stream_over_websocket(ws_url, args.prompt, max_tokens,
                                        temperature, use_spec)
        emit()
        return 0

    while True:
        try:
            emit()
            line = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            emit("\nbye.", style="dim")
            return 0

        if not line:
            continue

        if line.startswith("/"):
            command, _, rest = line.partition(" ")
            rest = rest.strip()
            if command in ("/quit", "/exit", "/q"):
                emit("bye.", style="dim")
                return 0
            if command == "/help":
                emit(HELP, style="dim")
            elif command == "/spec":
                use_spec = not use_spec
                emit(f"speculative decoding: {'on' if use_spec else 'off'}", style="cyan")
            elif command == "/tokens":
                try:
                    max_tokens = max(1, int(rest))
                    emit(f"max new tokens: {max_tokens}", style="cyan")
                except ValueError:
                    emit("usage: /tokens 64", style="red")
            elif command == "/temp":
                try:
                    temperature = float(rest)
                    emit(f"temperature: {temperature}", style="cyan")
                except ValueError:
                    emit("usage: /temp 0.7", style="red")
            elif command == "/stats":
                await show_json(base_url, "/v1/engine/telemetry", "engine telemetry")
            elif command == "/blocks":
                await show_json(base_url, "/v1/engine/blocks", "physical block map")
            elif command == "/metrics":
                await show_metrics(base_url)
            elif command == "/reset":
                history = []
                emit("conversation cleared.", style="cyan")
            else:
                emit(f"unknown command {command}; try /help", style="red")
            continue

        history.append({"role": "user", "content": line})
        emit("\nMemForge> ", style="bold green", end="")

        try:
            if args.sse:
                stats = await stream_over_sse(base_url, history, max_tokens,
                                              temperature, use_spec)
            else:
                # The WebSocket endpoint takes a raw prompt, so multi-turn
                # context is rendered here rather than by the chat template.
                multi_turn = len(history) > 1
                if multi_turn:
                    prompt = "\n".join(
                        f"{m['role']}: {m['content']}" for m in history
                    ) + "\nassistant:"
                else:
                    prompt = line
                stats = await stream_over_websocket(ws_url, prompt, max_tokens,
                                                    temperature, use_spec,
                                                    raw=multi_turn)
        except KeyboardInterrupt:
            # Deliberately exercises the server's disconnect path: the sequence
            # should be aborted and its blocks returned.
            emit("\n  interrupted; the server should abort the sequence and free its blocks",
                 style="yellow")
            history.pop()
            continue
        except Exception as exc:
            emit(f"\n  {type(exc).__name__}: {exc}", style="bold red")
            history.pop()
            continue

        if stats.error or not stats.text:
            history.pop()
        else:
            history.append({"role": "assistant", "content": stats.text})


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive MemForge chat client")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="server base URL")
    parser.add_argument("--tokens", type=int, default=64, help="max new tokens")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--spec", action="store_true", help="use speculative decoding")
    parser.add_argument("--sse", action="store_true",
                        help="use the OpenAI-compatible SSE path instead of the WebSocket")
    parser.add_argument("--prompt", help="send one prompt, print the answer, exit")
    args = parser.parse_args()

    try:
        return asyncio.run(repl(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
