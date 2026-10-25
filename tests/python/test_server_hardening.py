"""
Phase F server hardening: admission control, /metrics, and disconnect handling.

Kept in its own file rather than appended to `test_api_stream.py` so the two do
not collide on a shared branch.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from memforge_ml.runtime.continuous_engine import HAS_CPP_ENGINE, ContinuousBatchingEngine
from memforge_ml.runtime.sampling import SamplingParams
from server.api.disconnect_handler import ClientDisconnected, run_cancellable
from server.app import app
from server.engine_manager import ENGINE_MANAGER, EngineManager, EngineSaturated


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def stub_engine(tiny_model, stub_tokenizer, monkeypatch):
    if not HAS_CPP_ENGINE:
        pytest.skip("native memforge_c extension is not built")
    engine = ContinuousBatchingEngine(
        tiny_model,
        stub_tokenizer,
        block_size=4,
        num_blocks=64,
        max_num_seqs=4,
        max_model_len=128,
    )
    monkeypatch.setattr(ENGINE_MANAGER, "engine", engine)
    monkeypatch.setattr(ENGINE_MANAGER, "tokenizer", stub_tokenizer)
    monkeypatch.setattr(ENGINE_MANAGER, "ready", True)
    monkeypatch.setattr(ENGINE_MANAGER, "load_error", None)
    monkeypatch.setattr(ENGINE_MANAGER, "spec_engine", None)
    yield engine
    ENGINE_MANAGER._queues.clear()
    ENGINE_MANAGER._tracked.clear()
    ENGINE_MANAGER._in_flight = 0


# -------------------------------------------------------------------- metrics

def test_metrics_endpoint_is_prometheus_text(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")

    body = response.text
    # Always available, engine loaded or not.
    for metric in ("memforge_engine_ready", "memforge_requests_in_flight",
                   "memforge_max_concurrency"):
        assert f"# TYPE {metric} gauge" in body, metric
        assert any(line.startswith(metric + " ") for line in body.splitlines()), metric


def test_metrics_reports_engine_and_sharing_state(client, stub_engine):
    stub_engine.generate(["hello world"], SamplingParams(max_new_tokens=3, temperature=0.0))
    body = client.get("/metrics").text

    for metric in (
        "memforge_blocks_total",
        "memforge_blocks_allocated",
        "memforge_blocks_free",
        "memforge_blocks_shared",
        "memforge_blocks_saved",
        "memforge_memory_saved_mb",
        "memforge_prefix_hit_rate",
        "memforge_tokens_per_forward_pass",
        "memforge_preemptions_total",
        "memforge_cow_copies_total",
        "memforge_blocks_leaked",
    ):
        assert f"# TYPE {metric} gauge" in body, metric

    values = {
        line.split()[0]: line.split()[1]
        for line in body.splitlines()
        if line and not line.startswith("#")
    }
    # The invariant the stress harness watches.
    assert values["memforge_blocks_leaked"] == "0"
    assert int(values["memforge_blocks_total"]) == 64


def test_metrics_is_parseable_as_key_value_pairs(client, stub_engine):
    for line in client.get("/metrics").text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.partition(" ")
        assert name and value, line
        float(value)  # every metric value must be numeric


# ------------------------------------------------------------- backpressure

def test_engine_saturated_is_raised_at_the_limit(stub_engine, monkeypatch):
    """`max_concurrency` was dead config before this; prove it now bites."""
    monkeypatch.setattr(ENGINE_MANAGER.config, "max_concurrency", 2)

    async def scenario():
        params = SamplingParams(max_new_tokens=4, temperature=0.0)
        handles = [await ENGINE_MANAGER.submit(f"p{i}", params) for i in range(2)]
        assert ENGINE_MANAGER.in_flight == 2
        assert ENGINE_MANAGER.at_capacity

        with pytest.raises(EngineSaturated):
            await ENGINE_MANAGER.submit("one too many", params)

        # Capacity is returned when a stream closes, not leaked.
        ENGINE_MANAGER._close(handles[0].seq_id)
        assert ENGINE_MANAGER.in_flight == 1
        assert not ENGINE_MANAGER.at_capacity
        await ENGINE_MANAGER.submit("now there is room", params)
        assert ENGINE_MANAGER.in_flight == 2

    asyncio.run(scenario())


def test_saturation_returns_429_with_retry_after(client, stub_engine, monkeypatch):
    monkeypatch.setattr(ENGINE_MANAGER.config, "max_concurrency", 0)
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2},
    )
    assert response.status_code == 429
    assert response.headers.get("Retry-After") == "1"
    assert "capacity" in response.json()["detail"]


def test_saturation_is_429_not_503(client, stub_engine, monkeypatch):
    """
    EngineSaturated subclasses RuntimeError, which the handler maps to 503.
    Ordering the except clauses wrongly would turn "come back shortly" into
    "the engine is broken", so it is pinned by a test.
    """
    monkeypatch.setattr(ENGINE_MANAGER.config, "max_concurrency", 0)
    response = client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 2})
    assert response.status_code == 429


def test_in_flight_returns_to_zero_after_requests(client, stub_engine):
    for _ in range(3):
        response = client.post("/v1/completions", json={"prompt": "abc", "max_tokens": 2})
        assert response.status_code == 200
    assert ENGINE_MANAGER.in_flight == 0, "capacity leaked after completed requests"


def test_status_reports_admission_state(client, stub_engine):
    payload = client.get("/v1/engine/status").json()
    assert payload["in_flight"] == 0
    assert payload["at_capacity"] is False


# ---------------------------------------------------------- disconnect plumbing

def _fake_request(disconnected_after: int) -> Request:
    """A Request whose `is_disconnected()` flips to True after N calls."""
    calls = {"n": 0}

    class _FakeRequest:
        async def is_disconnected(self) -> bool:
            calls["n"] += 1
            return calls["n"] > disconnected_after

    return _FakeRequest()  # type: ignore[return-value]


def test_run_cancellable_returns_the_result_when_connected():
    async def work():
        await asyncio.sleep(0)
        return "done"

    result = asyncio.run(
        run_cancellable(_fake_request(disconnected_after=99), work(), poll_interval=0.01)
    )
    assert result == "done"


def test_run_cancellable_propagates_work_errors():
    async def work():
        raise ValueError("boom")

    async def scenario():
        with pytest.raises(ValueError, match="boom"):
            await run_cancellable(
                _fake_request(disconnected_after=99), work(), poll_interval=0.01
            )

    asyncio.run(scenario())


def test_run_cancellable_aborts_on_disconnect():
    """The Phase F gap: a non-streaming request must not outlive its client."""
    aborted = []

    async def never_finishes():
        await asyncio.sleep(30)

    async def scenario():
        with pytest.raises(ClientDisconnected):
            await run_cancellable(
                _fake_request(disconnected_after=0),
                never_finishes(),
                on_cancel=lambda: aborted.append(True),
                poll_interval=0.01,
            )

    asyncio.run(scenario())
    assert aborted == [True], "the engine was never told to abort"


def test_run_cancellable_cancels_the_work_on_disconnect():
    """Blocks must come back, which means the work really stops."""
    state = {"cancelled": False}

    async def work():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise

    async def scenario():
        with pytest.raises(ClientDisconnected):
            await run_cancellable(
                _fake_request(disconnected_after=0), work(), poll_interval=0.01
            )

    asyncio.run(scenario())
    assert state["cancelled"] is True


def test_on_cancel_failure_does_not_mask_the_disconnect():
    async def never_finishes():
        await asyncio.sleep(30)

    def exploding_cancel():
        raise RuntimeError("cleanup failed")

    async def scenario():
        with pytest.raises(ClientDisconnected):
            await run_cancellable(
                _fake_request(disconnected_after=0),
                never_finishes(),
                on_cancel=exploding_cancel,
                poll_interval=0.01,
            )

    asyncio.run(scenario())


def test_unreadable_transport_is_treated_as_connected():
    """
    A transport that cannot answer must degrade to the old behaviour, not abort
    a live request on a guess.
    """
    class _Broken:
        async def is_disconnected(self):
            raise RuntimeError("no receive channel")

    async def work():
        await asyncio.sleep(0.05)
        return "finished anyway"

    result = asyncio.run(run_cancellable(_Broken(), work(), poll_interval=0.01))  # type: ignore[arg-type]
    assert result == "finished anyway"


# ------------------------------------------------------- share-aware block map

def test_blocks_endpoint_exposes_block_owners(client, stub_engine):
    prompt = "the quick brown fox jumps over the lazy dog"
    params = SamplingParams(max_new_tokens=3, temperature=0.0)
    for i in range(3):
        stub_engine.submit(prompt + str(i), params)
    for _ in range(4):
        stub_engine.step()

    payload = client.get("/v1/engine/blocks").json()
    assert "occupancy" in payload              # original shape preserved
    assert "block_owners" in payload           # share-aware view added
    assert "shared_blocks" in payload
    assert isinstance(payload["block_owners"], dict)
    for block_str, owners in payload["block_owners"].items():
        assert payload["occupancy"][int(block_str)] in owners


# ------------------------------------------------------ threaded abort safety

def test_abort_is_deferred_to_the_stepping_thread(tiny_model, stub_tokenizer):
    """
    abort() must not tear the sequence down inline.

    The server aborts from its asyncio thread when a client disconnects, while
    step() runs in a worker thread. Tearing down inline raced with the step in
    progress: clearing rt.cache made a live decode fail with
    "'NoneType' object has no attribute 'indices'", and freeing the blocks
    mid-step was worse, because the step carried on writing into blocks the
    allocator had already reassigned.
    """
    engine = ContinuousBatchingEngine(
        tiny_model, stub_tokenizer, block_size=4, num_blocks=64,
        max_num_seqs=4, max_model_len=128,
    )
    seq_id = engine.submit("abort me", SamplingParams(max_new_tokens=50, temperature=0.0))
    engine.step()
    engine.step()

    runtime = engine.runtimes[seq_id]
    assert runtime.cache is not None
    allocated = engine.allocator.get_num_allocated_blocks()
    assert allocated > 0

    assert engine.abort(seq_id) is True
    # Nothing has been torn down yet: the request is queued, not applied.
    assert runtime.cache is not None, "abort mutated engine state inline"
    assert not runtime.done
    assert engine.allocator.get_num_allocated_blocks() == allocated
    # But the engine must know there is work left, or it would go idle with the
    # abort still pending.
    assert engine.has_work()

    engine.step()
    assert runtime.done
    assert runtime.finish_reason == "aborted"
    assert engine.allocator.get_num_free_blocks() == engine.allocator.get_total_blocks()


def test_concurrent_aborts_do_not_corrupt_a_running_step(tiny_model, stub_tokenizer):
    """
    Hammer abort() from other threads while the engine steps.

    This is the shape of the production failure: 32 of 96 requests errored under
    concurrent load with client disconnects. Any exception escaping step(), or
    any block left allocated at the end, is the bug returning.

    Honest caveat: this one is a smoke test, not a guarantee. Reinstating the
    inline abort does **not** reliably fail it, because hitting the window
    depends on thread timing. `test_abort_is_deferred_to_the_stepping_thread`
    above is the test with teeth -- it asserts the contract directly and does
    fail when the deferral is removed. This test is here to catch the coarser
    failures (an exception escaping, a leak) that a timing-independent test
    cannot see.
    """
    import threading

    engine = ContinuousBatchingEngine(
        tiny_model, stub_tokenizer, block_size=4, num_blocks=96,
        max_num_seqs=8, max_model_len=128,
    )
    params = SamplingParams(max_new_tokens=40, temperature=0.0, stop_on_eos=False)
    seq_ids = [engine.submit(f"concurrent abort {i}", params) for i in range(8)]

    # Let them all get admitted and prefilled first.
    for _ in range(3):
        engine.step()

    stop = threading.Event()
    errors = []

    def aborter(ids):
        while not stop.is_set():
            for seq_id in ids:
                try:
                    engine.abort(seq_id)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"abort raised {type(exc).__name__}: {exc}")
            time.sleep(0.001)

    threads = [
        threading.Thread(target=aborter, args=(seq_ids[:4],), daemon=True),
        threading.Thread(target=aborter, args=(seq_ids[4:],), daemon=True),
    ]
    for thread in threads:
        thread.start()

    try:
        steps = 0
        while engine.has_work() and steps < 400:
            try:
                engine.step()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"step raised {type(exc).__name__}: {exc}")
                break
            steps += 1
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=5)

    engine.step()

    assert not errors, errors
    total = engine.allocator.get_total_blocks()
    assert engine.allocator.get_num_free_blocks() == total, "blocks leaked under concurrent aborts"
    for seq_id in seq_ids:
        assert engine.runtimes[seq_id].done


def test_aborting_an_unknown_or_finished_sequence_is_false(tiny_model, stub_tokenizer):
    engine = ContinuousBatchingEngine(
        tiny_model, stub_tokenizer, block_size=4, num_blocks=32,
        max_num_seqs=2, max_model_len=64,
    )
    assert engine.abort(9999) is False

    seq_id = engine.submit("short", SamplingParams(max_new_tokens=2, temperature=0.0))
    engine.generate([])  # drain
    while engine.has_work():
        engine.step()
    engine.step()
    assert engine.abort(seq_id) is False, "a finished sequence cannot be aborted again"


# ------------------------------------------------- websocket prompt rendering

def test_ws_stream_applies_the_chat_template(client, stub_engine, monkeypatch):
    """
    /ws/stream must role-mark the prompt the way the HTTP path does.

    It used to pass the user's text straight into the model. For an
    instruction-tuned model that means no role markers, and SmolLM2-Instruct
    answers that by emitting EOS immediately -- one token, finish_reason
    "stop". The HTTP endpoint has always applied the template, so the two paths
    produced completely different output for the same input, and the dashboard
    streams over the WebSocket, so it got the broken one.
    """
    captured = {}
    original = ENGINE_MANAGER.stream

    def spy(prompt, params):
        captured["prompt"] = prompt
        return original(prompt, params)

    monkeypatch.setattr(ENGINE_MANAGER, "stream", spy)

    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"prompt": "hello there", "max_new_tokens": 2})
        while True:
            event = ws.receive_json()
            if event["type"] in ("done", "error"):
                break

    assert "prompt" in captured, "the engine was never asked to stream"
    sent = captured["prompt"]
    assert sent != "hello there", "the raw prompt reached the engine unrendered"
    assert "hello there" in sent
    # The stub tokenizer has no chat_template, so the fallback must still mark roles.
    assert "user" in sent and "assistant" in sent


def test_ws_stream_raw_flag_bypasses_the_template(client, stub_engine, monkeypatch):
    """`raw: true` is the escape hatch for callers formatting their own prompt."""
    captured = {}
    original = ENGINE_MANAGER.stream

    def spy(prompt, params):
        captured["prompt"] = prompt
        return original(prompt, params)

    monkeypatch.setattr(ENGINE_MANAGER, "stream", spy)

    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"prompt": "already formatted", "max_new_tokens": 2, "raw": True})
        while True:
            event = ws.receive_json()
            if event["type"] in ("done", "error"):
                break

    assert captured["prompt"] == "already formatted"


# ---------------------------------------------------- empty-prompt validation

@pytest.mark.parametrize(
    "url,body",
    [
        ("/v1/completions", {"prompt": "", "max_tokens": 2}),
        ("/v1/completions", {"prompt": "   \t ", "max_tokens": 2}),
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": ""}], "max_tokens": 2}),
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "  "}], "max_tokens": 2}),
        ("/v1/chat/completions", {"messages": [], "max_tokens": 2}),
    ],
)
def test_empty_prompts_are_rejected(client, stub_engine, url, body):
    """
    A prompt with nothing in it is a client error, not a generation request.

    All five of these used to return 200: an empty `prompt` tokenizes to a
    one-token sequence, and an empty chat message renders through the chat
    template into "user: \\nassistant:", which is not empty. So the engine
    generated from nothing and spent a scheduler slot and a block of KV cache
    doing it. 400 rather than 422 because the body is well-formed -- it is the
    content that is unusable.
    """
    response = client.post(url, json=body)
    assert response.status_code == 400, response.text
    assert "empty" in response.json()["detail"].lower()


def test_valid_prompts_still_pass(client, stub_engine):
    """The validation must not catch legitimate requests."""
    assert client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 2}).status_code == 200
    assert client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2},
    ).status_code == 200


def test_live_metrics_includes_sharing_keys(stub_engine):
    stub_engine.generate(["hello"], SamplingParams(max_new_tokens=2, temperature=0.0))
    metrics = ENGINE_MANAGER.live_metrics()
    for key in ("shared_blocks", "memory_saved_mb", "prefix_hit_rate",
                "in_flight", "max_concurrency"):
        assert key in metrics, key


# ------------------------------------- speculative stream concurrency safety

def test_speculative_worker_holds_a_thread_lock_for_its_whole_run():
    """
    The bug: abandoning the async generator released `_spec_lock` while the
    worker thread was still driving `PagedSpeculativeEngine`, which owns a
    single KV pool and a single cache. A second request then started a second
    worker against the same state.

    Reproduced directly against the real `speculative_stream`, with a stub
    engine whose `stream()` reports how many workers are inside it at once.
    """
    import threading

    concurrent = {"now": 0, "peak": 0}
    gate = threading.Lock()
    started = threading.Event()
    release = threading.Event()

    class _SlowSpecEngine:
        def stream(self, prompt, params):
            with gate:
                concurrent["now"] += 1
                concurrent["peak"] = max(concurrent["peak"], concurrent["now"])
            started.set()
            try:
                for i in range(100):
                    # Block until the test lets go, standing in for a long
                    # generation the client abandons part-way through.
                    release.wait(timeout=0.02)
                    yield type("E", (), {
                        "index": i, "text": "x", "token_id": 1,
                        "source": "draft", "round_index": 0,
                    })()
                return {}
            finally:
                with gate:
                    concurrent["now"] -= 1

    async def scenario():
        manager = EngineManager()
        manager.ready = True
        manager.spec_engine = _SlowSpecEngine()

        # First consumer takes one token then abandons the stream, exactly as a
        # disconnecting client does.
        first = manager.speculative_stream("p", SamplingParams(max_new_tokens=50))
        async for _ in first:
            break
        await first.aclose()

        assert started.is_set(), "the first worker never started"

        # Second request, immediately. If the lock were released early this
        # would run against the same engine as the still-live first worker.
        second = manager.speculative_stream("p", SamplingParams(max_new_tokens=50))
        async for _ in second:
            break
        await second.aclose()

        release.set()
        await asyncio.sleep(0.3)
        return concurrent["peak"]

    peak = asyncio.run(scenario())
    assert peak == 1, (
        f"{peak} workers were inside the speculative engine at once; the thread "
        "lock is not holding for the worker's full lifetime"
    )


def test_abandoned_speculative_stream_is_cancelled_early():
    """
    Cancellation must stop the worker, not merely let it finish.

    `stream()` frees the sequence's physical blocks in its own `finally`, so a
    worker that runs to completion does eventually clean up -- which is why
    asserting "the finally ran" proves nothing. What matters is that an
    abandoned request stops *promptly* instead of generating hundreds more
    tokens nobody will read, holding its blocks and the engine the whole time.

    So this counts what the generator actually produced: a few tokens with the
    cancel flag working, essentially all of them without it.
    """
    produced = {"count": 0}
    closed = threading.Event()
    TOTAL = 400

    class _CountingSpecEngine:
        def stream(self, prompt, params):
            try:
                for i in range(TOTAL):
                    produced["count"] += 1
                    yield type("E", (), {
                        "index": i, "text": "x", "token_id": 1,
                        "source": "target", "round_index": 0,
                    })()
                return {}
            finally:
                closed.set()

    async def scenario():
        manager = EngineManager()
        manager.ready = True
        manager.spec_engine = _CountingSpecEngine()
        stream = manager.speculative_stream("p", SamplingParams(max_new_tokens=TOTAL))
        async for _ in stream:
            break
        await stream.aclose()
        for _ in range(200):
            if closed.is_set():
                break
            await asyncio.sleep(0.01)

    asyncio.run(scenario())
    assert closed.is_set(), "stream()'s finally never ran, so blocks would leak"
    assert produced["count"] < TOTAL // 4, (
        f"the worker produced {produced['count']} of {TOTAL} tokens after the "
        "client left; cancellation is not reaching the generator"
    )
