"""
EngineManager: owns the models and drives the engine from the asyncio loop.

This is what turns the engine into a *server*. A single background task calls
`engine.step()` in a worker thread; every connected client's request is just
another sequence in that same loop. Two requests arriving 50 ms apart are
interleaved at the token level rather than queued behind one another -- which is
the whole point of continuous batching, and is only observable once requests come
from real concurrent clients.

Token events are fanned out to per-request asyncio queues, so HTTP and WebSocket
callers both consume the same stream.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional

import torch

from memforge_ml.runtime.continuous_engine import ContinuousBatchingEngine
from memforge_ml.runtime.sampling import SamplingParams
from memforge_ml.runtime.speculative_runtime import PagedSpeculativeEngine
from server.config import SERVER_CONFIG, ServerConfig

LOGGER = logging.getLogger(__name__)

DONE = object()  # sentinel pushed onto a request queue when its stream ends


class EngineSaturated(RuntimeError):
    """
    Raised when the engine is already serving `max_concurrency` requests.

    Surfaced to clients as 429 rather than queued behind a blocking await: a
    held-open connection on a saturated server turns load into timeouts, which
    is strictly worse than telling the caller to come back.
    """


@dataclass
class StreamHandle:
    seq_id: int
    queue: "asyncio.Queue[Any]"


class EngineManager:
    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or SERVER_CONFIG
        self.engine: Optional[ContinuousBatchingEngine] = None
        self.spec_engine: Optional[PagedSpeculativeEngine] = None
        self.tokenizer = None
        self.target_model = None
        self.draft_model = None

        self.load_error: Optional[str] = None
        self.ready = False
        self.loaded_at: Optional[float] = None

        self._queues: Dict[int, "asyncio.Queue[Any]"] = {}
        self._tracked: set = set()
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._last_report: Optional[Dict[str, Any]] = None

        # asyncio primitives bind to the loop that first awaits them, so they are
        # created per loop rather than in __init__. A long-lived manager can
        # legitimately see more than one loop (for example a test client that
        # spins up a fresh loop per request), and reusing primitives across loops
        # deadlocks.
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._wake = asyncio.Event()
        self._submit_lock = asyncio.Lock()
        self._spec_lock = asyncio.Lock()
        # A real thread lock, held by the speculative worker for its entire run.
        # The asyncio lock above only serialises *entering* the generator, and
        # an abandoned async generator releases it while the worker thread is
        # still driving the engine -- see speculative_stream() for why that
        # matters. Deliberately not rebuilt when the event loop changes: it
        # guards a thread, not a loop.
        self._spec_engine_lock = threading.Lock()
        # Admission control. `max_concurrency` has been in ServerConfig since the
        # server was written and was read by nothing -- a limit that looked
        # enforced but was not. Counted rather than a Semaphore so saturation can
        # be *reported* (429) instead of silently awaited.
        self._in_flight = 0

    # ----------------------------------------------------------- model loading

    def _load_blocking(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = self.config
        LOGGER.info("loading target model", extra={"model": cfg.target_model})
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.target_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.target_model = AutoModelForCausalLM.from_pretrained(
            cfg.target_model, dtype=dtype
        ).to(device)
        self.target_model.eval()

        # The draft model is loaded before the continuous engine rather than
        # after it, because with batched speculation enabled the engine needs it
        # at construction time. A failure here is not fatal: speculation is an
        # optimisation, so the engine is still built without it.
        if cfg.load_draft_model:
            try:
                LOGGER.info("loading draft model", extra={"model": cfg.draft_model})
                self.draft_model = AutoModelForCausalLM.from_pretrained(
                    cfg.draft_model, dtype=dtype
                ).to(device)
                self.draft_model.eval()
            except Exception as exc:  # draft model is optional
                LOGGER.warning(
                    "draft model unavailable, speculation disabled",
                    extra={"error": f"{type(exc).__name__}: {exc}"},
                )
                self.draft_model = None

        speculate_in_batch = bool(cfg.batched_speculation) and self.draft_model is not None
        if cfg.batched_speculation and self.draft_model is None:
            LOGGER.warning(
                "batched speculation requested but no draft model is loaded; "
                "continuing without it"
            )

        self.engine = ContinuousBatchingEngine(
            self.target_model,
            self.tokenizer,
            block_size=cfg.block_size,
            kv_budget_mb=cfg.kv_budget_mb,
            max_num_seqs=cfg.max_num_seqs,
            max_model_len=cfg.max_model_len,
            batched_decode=cfg.batched_decode,
            decode_view_cache=cfg.decode_view_cache,
            view_cache_budget_mb=cfg.view_cache_budget_mb,
            block_attention=cfg.block_attention,
            max_prefill_tokens_per_step=cfg.max_prefill_tokens_per_step,
            draft_model=self.draft_model if speculate_in_batch else None,
            lookahead_k=cfg.lookahead_k if speculate_in_batch else 0,
        )
        pool = self.engine.pool.describe()
        LOGGER.info(
            "continuous engine ready",
            extra={
                "kv_pool_mb": pool["total_mb"],
                "num_blocks": pool["num_blocks"],
                "block_size": pool["block_size"],
                "capacity_tokens": pool["capacity_tokens"],
                "device": pool["device"],
                "batched_speculation": self.engine.speculation_enabled,
                "lookahead_k": self.engine.lookahead_k,
            },
        )

        # The standalone single-sequence speculative engine, still served on its
        # own endpoint. It is separate from the engine above on purpose: this one
        # is one sequence at a time behind a lock, which is what Phase D's
        # batched path replaces rather than extends.
        if self.draft_model is not None:
            try:
                self.spec_engine = PagedSpeculativeEngine(
                    self.target_model,
                    self.tokenizer,
                    self.draft_model,
                    self.tokenizer,
                    lookahead_k=cfg.lookahead_k,
                    block_size=cfg.block_size,
                    kv_budget_mb=cfg.kv_budget_mb / 2,
                    max_model_len=cfg.max_model_len,
                )
                LOGGER.info("speculative engine ready")
            except Exception as exc:  # the standalone path is optional
                LOGGER.warning(
                    "standalone speculative engine unavailable",
                    extra={"error": f"{type(exc).__name__}: {exc}"},
                )
                self.spec_engine = None

    async def start(self) -> None:
        self._running = True
        if self.config.eager_load:
            await self.ensure_loaded()
        self._ensure_loop()

    async def ensure_loaded(self) -> bool:
        if self.ready:
            return True
        if self.load_error is not None:
            return False
        try:
            await asyncio.get_running_loop().run_in_executor(None, self._load_blocking)
            self.ready = True
            self.loaded_at = time.time()
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            LOGGER.error("model load failed", extra={"error": self.load_error})
            return False
        return True

    async def stop(self) -> None:
        self._running = False
        self._wake.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        for queue in self._queues.values():
            queue.put_nowait(DONE)
        self._queues.clear()
        self._tracked.clear()
        self._in_flight = 0

    # ------------------------------------------------------------- engine loop

    async def _engine_loop(self) -> None:
        """Drive the engine one step at a time, forever, fanning out tokens."""
        loop = asyncio.get_running_loop()
        while self._running:
            if self.engine is None or not self.engine.has_work():
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    pass
                continue

            try:
                report = await loop.run_in_executor(None, self.engine.step)
            except Exception as exc:  # never let one bad step kill the server
                LOGGER.error(
                    "engine step failed",
                    extra={"error": f"{type(exc).__name__}: {exc}",
                           "tracked_requests": len(self._tracked)},
                    exc_info=True,
                )
                for seq_id in list(self._tracked):
                    self._emit(seq_id, {"type": "error", "message": str(exc)})
                    self._close(seq_id)
                continue

            self._last_report = {
                "step": report.step_index,
                "running": report.num_running,
                "waiting": report.num_waiting,
                "prefill": report.num_prefill,
                "decode": report.num_decode,
                "free_blocks": report.free_blocks,
                "total_blocks": report.total_blocks,
                "memory_utilization": report.memory_utilization,
                "step_latency_ms": round(report.step_latency_ms, 2),
                "preempted": list(report.preempted),
            }

            for event in report.events:
                self._emit(
                    event.seq_id,
                    {
                        "type": "token",
                        "seq_id": event.seq_id,
                        "index": event.index,
                        "token": event.text,
                        "token_id": event.token_id,
                        "source": event.source,
                        "is_draft": event.source == "draft",
                        "metrics": self.live_metrics(),
                    },
                )

            for seq_id in list(self._tracked):
                runtime = self.engine.runtimes.get(seq_id)
                if runtime is not None and runtime.done:
                    result = self.engine.result_for(seq_id)
                    self._emit(seq_id, {"type": "done", **result, "metrics": self.live_metrics()})
                    self._close(seq_id)

    def _emit(self, seq_id: int, payload: Dict[str, Any]) -> None:
        queue = self._queues.get(seq_id)
        if queue is not None:
            queue.put_nowait(payload)

    def _close(self, seq_id: int) -> None:
        queue = self._queues.pop(seq_id, None)
        was_tracked = seq_id in self._tracked
        self._tracked.discard(seq_id)
        if was_tracked and self._in_flight > 0:
            # Every stream ends here exactly once -- completion, abort, error and
            # shutdown all funnel through _close -- so the in-flight count cannot
            # drift and permanently strand capacity.
            self._in_flight -= 1
        if queue is not None:
            queue.put_nowait(DONE)

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def at_capacity(self) -> bool:
        return self._in_flight >= self.config.max_concurrency

    # -------------------------------------------------------------- submission

    def _ensure_loop(self) -> None:
        """
        Make sure the stepping task is running on the current event loop.

        Requests depend on this loop, so it must exist even when the app was
        created without its lifespan (as a bare `TestClient(app)` does) or if the
        task died unexpectedly. If the running loop has changed since last time,
        the loop-bound primitives are rebuilt before restarting the task.
        """
        running_loop = asyncio.get_running_loop()
        if self._loop is not running_loop:
            self._loop = running_loop
            self._wake = asyncio.Event()
            self._submit_lock = asyncio.Lock()
            self._spec_lock = asyncio.Lock()
            # Queues belonged to the previous loop and can never be drained now.
            self._queues.clear()
            self._tracked.clear()
            self._in_flight = 0
            self._task = None

        if self._task is None or self._task.done():
            self._running = True
            self._task = asyncio.create_task(self._engine_loop())

    async def submit(self, prompt: str, params: SamplingParams) -> StreamHandle:
        if not await self.ensure_loaded():
            raise RuntimeError(f"engine unavailable: {self.load_error}")
        self._ensure_loop()

        async with self._submit_lock:
            # Checked under the lock so simultaneous arrivals cannot both slip
            # past the limit.
            if self._in_flight >= self.config.max_concurrency:
                raise EngineSaturated(
                    f"engine at capacity: {self._in_flight} requests in flight "
                    f"(max_concurrency={self.config.max_concurrency})"
                )
            seq_id = self.engine.submit(prompt, params)
            queue: "asyncio.Queue[Any]" = asyncio.Queue()
            self._queues[seq_id] = queue
            self._tracked.add(seq_id)
            self._in_flight += 1
        self._wake.set()
        return StreamHandle(seq_id=seq_id, queue=queue)

    async def stream(self, prompt: str, params: SamplingParams) -> AsyncGenerator[Dict[str, Any], None]:
        handle = await self.submit(prompt, params)
        try:
            while True:
                item = await handle.queue.get()
                if item is DONE:
                    break
                yield item
        finally:
            if handle.seq_id in self._tracked:
                self.engine.abort(handle.seq_id)
                self._close(handle.seq_id)

    async def generate(self, prompt: str, params: SamplingParams) -> Dict[str, Any]:
        """Non-streaming generation through the shared continuous-batching engine."""
        handle = await self.submit(prompt, params)
        final: Optional[Dict[str, Any]] = None
        while True:
            item = await handle.queue.get()
            if item is DONE:
                break
            if item.get("type") == "done":
                final = item
            elif item.get("type") == "error":
                raise RuntimeError(item["message"])
        if final is None:
            raise RuntimeError("generation ended without producing a result")
        return final

    # ------------------------------------------------------------- speculative

    async def speculative_generate(self, prompt: str, params: SamplingParams) -> Dict[str, Any]:
        if not await self.ensure_loaded():
            raise RuntimeError(f"engine unavailable: {self.load_error}")
        if self.spec_engine is None:
            raise RuntimeError("speculative decoding is not available (no draft model loaded)")
        self._ensure_loop()
        loop = asyncio.get_running_loop()
        # The speculative engine holds a single KV pool, so serialize access.
        async with self._spec_lock:
            return await loop.run_in_executor(
                None, lambda: self.spec_engine.generate(prompt, params)
            )

    async def speculative_stream(
        self, prompt: str, params: SamplingParams
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream speculative tokens, tagging each as draft-accepted or target-corrected."""
        if not await self.ensure_loaded():
            raise RuntimeError(f"engine unavailable: {self.load_error}")
        if self.spec_engine is None:
            raise RuntimeError("speculative decoding is not available (no draft model loaded)")

        self._ensure_loop()
        loop = asyncio.get_running_loop()
        queue: "asyncio.Queue[Any]" = asyncio.Queue()
        cancelled = threading.Event()

        def worker() -> None:
            # Held for the whole run, not just while the consumer is listening.
            # `PagedSpeculativeEngine` owns a single KV pool and a single cache,
            # so two workers driving it at once corrupt each other's state.
            with self._spec_engine_lock:
                generator = None
                try:
                    generator = self.spec_engine.stream(prompt, params)
                    while True:
                        if cancelled.is_set():
                            break
                        try:
                            event = next(generator)
                        except StopIteration as stop:
                            loop.call_soon_threadsafe(
                                queue.put_nowait, {"type": "done", **(stop.value or {})}
                            )
                            break
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            {
                                "type": "token",
                                "index": event.index,
                                "token": event.text,
                                "token_id": event.token_id,
                                "source": event.source,
                                "is_draft": event.source == "draft",
                                "round": event.round_index,
                            },
                        )
                except Exception as exc:
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        {"type": "error", "message": f"{type(exc).__name__}: {exc}"},
                    )
                finally:
                    # close() raises GeneratorExit inside stream(), whose own
                    # `finally` frees the sequence's physical blocks. Without
                    # this an abandoned speculative stream leaks its blocks for
                    # the life of the process.
                    if generator is not None:
                        try:
                            generator.close()
                        except Exception:
                            pass
                    loop.call_soon_threadsafe(queue.put_nowait, DONE)

        async with self._spec_lock:
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            try:
                while True:
                    item = await queue.get()
                    if item is DONE:
                        break
                    yield item
            finally:
                # Reached on a client disconnect, which closes this generator.
                # Only a flag is set: awaiting a thread join here is illegal
                # during GeneratorExit, and the worker holds
                # _spec_engine_lock until it actually stops, so the next
                # request waits for the engine rather than racing it.
                cancelled.set()

    # ---------------------------------------------------------------- telemetry

    def live_metrics(self) -> Dict[str, Any]:
        if self.engine is None:
            return {}
        telemetry = self.engine.telemetry()
        return {
            "tokens_per_second": telemetry["tokens_per_second"],
            "tokens_per_forward_pass": telemetry["tokens_per_forward_pass"],
            "allocated_blocks": telemetry["allocated_blocks"],
            "free_blocks": telemetry["free_blocks"],
            "total_blocks": telemetry["total_blocks"],
            "memory_utilization": telemetry["memory_utilization"],
            "running_sequences": telemetry["running_sequences"],
            "waiting_sequences": telemetry["waiting_sequences"],
            "preemptions": telemetry["preemptions"],
            # Prefix sharing. Present and zeroed when sharing is off, so the
            # dashboard binds to one stable shape either way.
            "shared_blocks": telemetry["shared_blocks"],
            "memory_saved_mb": telemetry["memory_saved_mb"],
            "prefix_hit_rate": telemetry["prefix_hit_rate"],
            "in_flight": self._in_flight,
            "max_concurrency": self.config.max_concurrency,
        }

    def status(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "ready": self.ready,
            "load_error": self.load_error,
            "speculative_available": self.spec_engine is not None,
            "in_flight": self._in_flight,
            "at_capacity": self.at_capacity,
            "config": self.config.model_dump(),
        }
        if self.engine is not None:
            payload["engine"] = self.engine.describe()
            payload["telemetry"] = self.engine.telemetry()
            payload["last_step"] = self._last_report
        return payload

    def block_occupancy(self) -> List[int]:
        return self.engine.block_occupancy() if self.engine is not None else []

    def block_owners(self) -> Dict[str, List[int]]:
        """Share-aware block map: block id -> every sequence referencing it."""
        return self.engine.block_owners() if self.engine is not None else {}

    def prometheus_metrics(self) -> str:
        """
        Plain-text Prometheus exposition.

        Hand-rolled rather than pulling in a client library: the whole surface is
        a dozen gauges. It doubles as the scrape target for the stress harness,
        so that does not have to track the JSON telemetry shape.
        """
        lines = [
            "# HELP memforge_engine_ready Whether the engine finished loading.",
            "# TYPE memforge_engine_ready gauge",
            f"memforge_engine_ready {int(self.ready)}",
            "# HELP memforge_requests_in_flight Requests currently admitted.",
            "# TYPE memforge_requests_in_flight gauge",
            f"memforge_requests_in_flight {self._in_flight}",
            "# HELP memforge_max_concurrency Configured admission limit.",
            "# TYPE memforge_max_concurrency gauge",
            f"memforge_max_concurrency {self.config.max_concurrency}",
        ]
        if self.engine is None:
            return "\n".join(lines) + "\n"

        t = self.engine.telemetry()
        gauges = [
            ("memforge_blocks_total", t["total_blocks"], "Physical KV blocks in the pool."),
            ("memforge_blocks_allocated", t["allocated_blocks"], "Blocks currently held."),
            ("memforge_blocks_free", t["free_blocks"], "Blocks available."),
            ("memforge_blocks_shared", t["shared_blocks"], "Blocks with more than one referrer."),
            ("memforge_blocks_saved", t["blocks_saved"], "Allocations avoided by prefix sharing."),
            ("memforge_memory_saved_mb", t["memory_saved_mb"], "KV megabytes saved by sharing."),
            ("memforge_prefix_hit_rate", t["prefix_hit_rate"], "Prefix-cache hit rate."),
            ("memforge_prefix_cached_blocks", t["cached_prefix_blocks"], "Blocks registered shareable."),
            ("memforge_memory_utilization", t["memory_utilization"], "Fraction of the pool allocated."),
            ("memforge_sequences_running", t["running_sequences"], "Sequences in the running queue."),
            ("memforge_sequences_waiting", t["waiting_sequences"], "Sequences awaiting admission."),
            ("memforge_tokens_per_forward_pass", t["tokens_per_forward_pass"], "Batching effectiveness."),
            ("memforge_tokens_generated_total", t["tokens_generated"], "Tokens generated since start."),
            ("memforge_forward_passes_total", t["forward_passes"], "Forward passes since start."),
            ("memforge_preemptions_total", t["preemptions"], "Sequences preempted under pressure."),
            ("memforge_cow_copies_total", t["cow_copies"], "Copy-on-write block copies performed."),
        ]
        for name, value, help_text in gauges:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")

        # The invariant the stress harness watches: nothing leaks.
        leaked = t["total_blocks"] - t["free_blocks"] - t["allocated_blocks"]
        lines.append("# HELP memforge_blocks_leaked total - free - allocated; must always be 0.")
        lines.append("# TYPE memforge_blocks_leaked gauge")
        lines.append(f"memforge_blocks_leaked {leaked}")
        return "\n".join(lines) + "\n"


ENGINE_MANAGER = EngineManager()
