"""
Phase F server hardening: admission control, /metrics, and disconnect handling.

Kept in its own file rather than appended to `test_api_stream.py` so the two do
not collide on a shared branch.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from memforge_ml.runtime.continuous_engine import HAS_CPP_ENGINE, ContinuousBatchingEngine
from memforge_ml.runtime.sampling import SamplingParams
from server.api.disconnect_handler import ClientDisconnected, run_cancellable
from server.app import app
from server.engine_manager import ENGINE_MANAGER, EngineSaturated


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
