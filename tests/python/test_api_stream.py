"""
API tests.

The endpoint tests that need generation patch a small stub engine into the
EngineManager rather than loading real weights, so the suite stays fast and
offline while still exercising the real routing, streaming and error paths.
"""

import json

import pytest
from fastapi.testclient import TestClient

from memforge_ml.runtime.continuous_engine import HAS_CPP_ENGINE, ContinuousBatchingEngine
from server.app import app
from server.engine_manager import ENGINE_MANAGER


@pytest.fixture
def client():
    # The context manager runs the app lifespan and keeps every request on one
    # event loop, which is what a real server does.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def stub_engine(tiny_model, stub_tokenizer, monkeypatch):
    """Point the shared EngineManager at a tiny in-memory engine."""
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


# ------------------------------------------------------------------- metadata

def test_health_check_endpoint(client):
    response = client.get("/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] in {"healthy", "degraded"}
    assert "MemForge" in data["service"]
    assert "engine_ready" in data


def test_models_endpoint(client):
    response = client.get("/v1/models")

    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) >= 1
    assert data["data"][0]["id"] == "memforge-target"


def test_status_endpoint_reports_configuration(client):
    response = client.get("/v1/engine/status")

    assert response.status_code == 200
    data = response.json()
    assert "config" in data
    assert "ready" in data


def test_telemetry_endpoint_returns_503_before_the_engine_loads(client, monkeypatch):
    monkeypatch.setattr(ENGINE_MANAGER, "engine", None)

    response = client.get("/v1/engine/telemetry")

    assert response.status_code == 503


def test_telemetry_endpoint_reports_pool_state(client, stub_engine):
    response = client.get("/v1/engine/telemetry")

    assert response.status_code == 200
    data = response.json()
    assert data["total_blocks"] == 64
    assert data["block_size"] == 4
    assert data["free_blocks"] + data["allocated_blocks"] == data["total_blocks"]


def test_blocks_endpoint_returns_an_occupancy_map(client, stub_engine):
    response = client.get("/v1/engine/blocks")

    assert response.status_code == 200
    data = response.json()
    assert len(data["occupancy"]) == data["total_blocks"] == 64
    assert data["block_size"] == 4


# ----------------------------------------------------------------- generation

def test_chat_completions_endpoint(client, stub_engine):
    payload = {
        "messages": [{"role": "user", "content": "what is paged attention"}],
        "max_new_tokens": 8,
        "temperature": 0.0,
        "top_p": 1.0,
        "use_speculative": False,
    }

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(data["choices"][0]["message"]["content"], str)
    assert data["usage"]["completion_tokens"] == 8
    assert data["usage"]["total_tokens"] == (
        data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]
    )
    assert data["memforge_telemetry"]["mode"] == "continuous_batching"


def test_completions_endpoint(client, stub_engine):
    response = client.post(
        "/v1/completions",
        json={"prompt": "raw prompt", "max_tokens": 6, "temperature": 0.0, "top_p": 1.0},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["generated_tokens"] == 6
    assert data["finish_reason"] == "length"


def test_chat_completions_streams_sse_chunks(client, stub_engine):
    payload = {
        "messages": [{"role": "user", "content": "stream please"}],
        "max_new_tokens": 6,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
    }

    with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        assert response.status_code == 200
        chunks = [
            json.loads(line[6:])
            for line in response.iter_lines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]

    assert chunks, "no SSE chunks received"
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    content = "".join(
        c["choices"][0]["delta"].get("content", "")
        for c in chunks
        if c["choices"][0]["delta"].get("content")
    )
    assert content
    assert chunks[-1]["choices"][0]["finish_reason"] is not None


def test_concurrent_requests_are_interleaved_by_the_scheduler(client, stub_engine):
    """
    Several requests in flight at once must all be served, and the engine should
    confirm more than one token per forward pass -- the observable signature of
    continuous batching.
    """
    import threading

    results = {}

    def fire(index):
        response = client.post(
            "/v1/completions",
            json={
                "prompt": f"concurrent prompt {index}",
                "max_tokens": 8,
                "temperature": 0.0,
                "top_p": 1.0,
            },
        )
        results[index] = response

    threads = [threading.Thread(target=fire, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert len(results) == 4
    for response in results.values():
        assert response.status_code == 200
        assert response.json()["generated_tokens"] == 8
    assert stub_engine.telemetry()["tokens_per_forward_pass"] > 1.0


def test_speculative_request_is_rejected_when_unavailable(client, stub_engine):
    payload = {
        "messages": [{"role": "user", "content": "speculate"}],
        "max_new_tokens": 4,
        "use_speculative": True,
    }

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 503
    assert "speculative" in response.json()["detail"].lower()


def test_oversized_prompt_returns_400(client, stub_engine):
    payload = {
        "messages": [{"role": "user", "content": "x" * 400}],
        "max_new_tokens": 4,
        "temperature": 0.0,
    }

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 400


# ------------------------------------------------------------------ websockets

def test_websocket_stream_emits_tokens_then_done(client, stub_engine):
    with client.websocket_connect("/ws/stream") as websocket:
        websocket.send_text(
            json.dumps(
                {
                    "prompt": "websocket prompt",
                    "max_new_tokens": 6,
                    "temperature": 0.0,
                    "top_p": 1.0,
                }
            )
        )

        start = websocket.receive_json()
        assert start["type"] == "start"
        assert start["mode"] == "continuous_batching"

        tokens, done = [], None
        while True:
            event = websocket.receive_json()
            if event["type"] == "token":
                tokens.append(event)
            elif event["type"] == "done":
                done = event
                break
            elif event["type"] == "error":
                pytest.fail(f"stream errored: {event['message']}")

    assert len(tokens) == 6
    assert [t["index"] for t in tokens] == list(range(6))
    assert all("source" in t and "metrics" in t for t in tokens)
    assert done["generated_tokens"] == 6


def test_websocket_rejects_an_empty_prompt(client, stub_engine):
    with client.websocket_connect("/ws/stream") as websocket:
        websocket.send_text(json.dumps({"prompt": "   "}))
        event = websocket.receive_json()

    assert event["type"] == "error"
    assert "required" in event["message"]


def test_websocket_telemetry_feed_pushes_engine_state(client, stub_engine):
    with client.websocket_connect("/ws/telemetry") as websocket:
        event = websocket.receive_json()

    assert event["type"] == "telemetry"
    assert event["telemetry"]["total_blocks"] == 64
    assert len(event["occupancy"]) == 64
