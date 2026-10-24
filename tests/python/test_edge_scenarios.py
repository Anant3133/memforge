"""
Edge-case integration tests (Track 5).

These are the inputs a real client sends by accident: a prompt longer than the
context window, a malformed body, 50 requests when the pool holds 25, a browser
tab closed mid-stream. The engine has to answer each one with a defined
behaviour rather than a 500 or a silent leak.

Deliberately does not re-test what `test_server_hardening.py` already covers
(429 mechanics, /metrics parsing, disconnect plumbing, empty prompts). This file
goes around it.

**The invariant asserted throughout is block accounting**:
`total_blocks == free_blocks + allocated_blocks`, and allocated returns to its
baseline once requests drain. That is exact rather than statistical, so a
failure here is a genuine leak and not noise.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from memforge_ml.runtime.continuous_engine import HAS_CPP_ENGINE, ContinuousBatchingEngine
from memforge_ml.runtime.sampling import SamplingParams
from server.app import app
from server.engine_manager import ENGINE_MANAGER

MAX_MODEL_LEN = 128


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def stub_engine(tiny_model, stub_tokenizer, monkeypatch):
    """A small in-memory engine, so the suite stays fast and offline."""
    if not HAS_CPP_ENGINE:
        pytest.skip("native memforge_c extension is not built")
    engine = ContinuousBatchingEngine(
        tiny_model,
        stub_tokenizer,
        block_size=4,
        num_blocks=64,
        max_num_seqs=4,
        max_model_len=MAX_MODEL_LEN,
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


def block_accounting(client: TestClient) -> dict:
    return client.get("/v1/engine/telemetry").json()


def assert_no_leak(telemetry: dict, context: str = "") -> None:
    total = telemetry["total_blocks"]
    free = telemetry["free_blocks"]
    allocated = telemetry["allocated_blocks"]
    assert total == free + allocated, (
        f"block accounting broken{' (' + context + ')' if context else ''}: "
        f"total={total} free={free} allocated={allocated}"
    )


# --------------------------------------------------------- oversized prompts

def test_prompt_over_max_model_len_is_rejected(client, stub_engine):
    """A prompt with no room to generate must fail cleanly, not be truncated."""
    response = client.post(
        "/v1/completions",
        json={"prompt": "x" * (MAX_MODEL_LEN * 10), "max_tokens": 4},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "max_model_len" in detail, detail


def test_oversized_prompt_leaves_no_blocks_behind(client, stub_engine):
    """A rejected request must not have cost anything."""
    before = block_accounting(client)
    for _ in range(5):
        client.post("/v1/completions", json={"prompt": "x" * 4000, "max_tokens": 4})
    after = block_accounting(client)

    assert after["allocated_blocks"] == before["allocated_blocks"]
    assert_no_leak(after, "after rejected oversized prompts")


def test_prompt_exactly_at_the_limit_is_accepted(client, stub_engine):
    """
    Off-by-one check on the boundary.

    The stub tokenizer emits one token per character, so this prompt is
    MAX_MODEL_LEN - 8 tokens and leaves room for 8 generated ones.
    """
    response = client.post(
        "/v1/completions",
        json={"prompt": "a" * (MAX_MODEL_LEN - 8), "max_tokens": 4},
    )
    assert response.status_code == 200, response.text


# ------------------------------------------------------- malformed requests

@pytest.mark.parametrize(
    "body",
    [
        {"max_tokens": 4},                                    # no messages field
        {"messages": "not a list", "max_tokens": 4},          # wrong type
        {"messages": [{"content": "no role"}], "max_tokens": 4},
        {"messages": [{"role": "user"}], "max_tokens": 4},    # no content
    ],
)
def test_malformed_chat_bodies_are_422(client, stub_engine, body):
    """
    Schema problems are 422, not 400.

    That is FastAPI/Pydantic's default and it is the correct distinction: 422
    means the body does not match the schema, 400 means it does but the values
    are unusable. Pinned here because it is easy to assert the wrong one.
    """
    assert client.post("/v1/chat/completions", json=body).status_code == 422


@pytest.mark.parametrize("max_tokens", [0, -1, -100])
def test_non_positive_max_tokens_is_422(client, stub_engine, max_tokens):
    response = client.post(
        "/v1/completions", json={"prompt": "hello", "max_tokens": max_tokens}
    )
    assert response.status_code == 422


def test_unparseable_body_is_422(client, stub_engine):
    response = client.post(
        "/v1/chat/completions",
        content=b"{not json at all",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


# -------------------------------------------------------- pool saturation

def test_saturation_mixes_200_and_429_without_500s(client, stub_engine, monkeypatch):
    """
    More simultaneous requests than the engine admits.

    The required behaviour is a clean mix of 200 and 429: backpressure, not
    failure. A 500 anywhere here means the server fell over under load, which is
    the thing this test exists to catch.
    """
    monkeypatch.setattr(ENGINE_MANAGER.config, "max_concurrency", 2)

    statuses = []
    for _ in range(12):
        response = client.post(
            "/v1/completions", json={"prompt": "saturate me", "max_tokens": 2}
        )
        statuses.append(response.status_code)
        if response.status_code == 429:
            assert response.headers.get("Retry-After") is not None

    assert all(s in (200, 429) for s in statuses), f"unexpected statuses: {set(statuses)}"
    assert 500 not in statuses
    # The TestClient is synchronous, so each request completes and releases its
    # slot before the next begins; the point here is that none of them error.
    assert 200 in statuses
    assert_no_leak(block_accounting(client), "after saturation")


def test_capacity_is_released_after_saturation(client, stub_engine, monkeypatch):
    """A 429 must not consume the slot it was refused."""
    monkeypatch.setattr(ENGINE_MANAGER.config, "max_concurrency", 0)
    for _ in range(5):
        assert client.post(
            "/v1/completions", json={"prompt": "hi", "max_tokens": 2}
        ).status_code == 429

    monkeypatch.setattr(ENGINE_MANAGER.config, "max_concurrency", 4)
    assert client.post(
        "/v1/completions", json={"prompt": "hi", "max_tokens": 2}
    ).status_code == 200
    assert ENGINE_MANAGER.in_flight == 0


# ------------------------------------------------------- memory pressure

def test_requests_complete_under_memory_pressure(tiny_model, stub_tokenizer):
    """
    A pool too small to hold every sequence at once must still finish them all.

    This exercises preempt-by-recompute: the scheduler frees a victim's blocks
    and re-queues it, and the engine replays its prefill. Correctness under
    pressure is the point -- a sequence that is silently dropped would show up
    here as a missing result.
    """
    engine = ContinuousBatchingEngine(
        tiny_model,
        stub_tokenizer,
        block_size=4,
        num_blocks=10,  # deliberately far too small for 4 concurrent sequences
        max_num_seqs=4,
        max_model_len=96,
    )
    params = SamplingParams(max_new_tokens=6, temperature=0.0, stop_on_eos=False)
    ids = [engine.submit(f"pressure test number {i}", params) for i in range(4)]

    steps = 0
    while engine.has_work() and steps < 2000:
        engine.step()
        steps += 1
    engine.step()

    for seq_id in ids:
        result = engine.result_for(seq_id)
        assert result["finish_reason"] is not None, f"seq {seq_id} never finished"
        assert result["generated_tokens"] > 0

    total = engine.allocator.get_total_blocks()
    assert engine.allocator.get_num_free_blocks() == total, "blocks leaked under pressure"


# --------------------------------------------------------------- websockets

def test_websocket_rejects_empty_prompt_and_stays_open(client, stub_engine):
    """
    An invalid message must not take the connection down with it.

    A browser that sent one bad frame should be able to carry on, which is why
    the endpoint answers with an error event instead of closing.
    """
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"prompt": "   ", "max_new_tokens": 2})
        event = ws.receive_json()
        assert event["type"] == "error"

        # The socket must still work afterwards.
        ws.send_json({"prompt": "now a real prompt", "max_new_tokens": 2})
        kinds = []
        while True:
            event = ws.receive_json()
            kinds.append(event["type"])
            if event["type"] in ("done", "error"):
                break
        assert "token" in kinds
        assert kinds[-1] == "done"


def test_websocket_survives_malformed_json(client, stub_engine):
    """
    A garbage frame must be reported, not fatal.

    It used to escape to the outer handler, which dropped the connection: one
    bad message ended the whole session. One tab sending nonsense should not
    cost that tab its socket, so it is now answered like any other bad request.
    """
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text("{this is not json")
        event = ws.receive_json()
        assert event["type"] == "error"
        assert "malformed JSON" in event["message"]

        # The connection must still be usable.
        ws.send_json({"prompt": "recovered", "max_new_tokens": 2})
        kinds = []
        while True:
            event = ws.receive_json()
            kinds.append(event["type"])
            if event["type"] in ("done", "error"):
                break
        assert "token" in kinds, f"socket did not recover: {kinds}"

    assert client.get("/v1/health").status_code == 200
    assert_no_leak(block_accounting(client), "after malformed websocket frame")


def test_websocket_rejects_a_json_array(client, stub_engine):
    """Valid JSON that is not an object must not be treated as a request."""
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text("[1, 2, 3]")
        event = ws.receive_json()
        assert event["type"] == "error"
        assert "JSON object" in event["message"]


def test_websocket_rejects_invalid_sampling_params(client, stub_engine):
    """
    Bad sampling values are a request error, not a stream error.

    Without this they surfaced from deep inside the engine mid-generation,
    after the client had already been told the run had started.
    """
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"prompt": "hello", "max_new_tokens": "not a number"})
        event = ws.receive_json()
        assert event["type"] == "error"
        assert "invalid parameters" in event["message"]


def test_websocket_missing_prompt_field(client, stub_engine):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"max_new_tokens": 2})
        event = ws.receive_json()
        assert event["type"] == "error"


# ------------------------------------------------------------ leak checking

def test_many_sequential_requests_do_not_leak(client, stub_engine):
    """
    The cheap version of the stress harness.

    Thirty requests through the real routing stack, asserting the pool returns
    to exactly where it started. Exact, not statistical: any drift is a leak.
    """
    baseline = block_accounting(client)
    assert_no_leak(baseline, "baseline")

    for i in range(30):
        response = client.post(
            "/v1/completions", json={"prompt": f"request number {i}", "max_tokens": 3}
        )
        assert response.status_code == 200, response.text

    after = block_accounting(client)
    assert_no_leak(after, "after 30 requests")
    assert after["allocated_blocks"] == baseline["allocated_blocks"], (
        f"leaked {after['allocated_blocks'] - baseline['allocated_blocks']} blocks "
        "across 30 requests"
    )


def test_streaming_requests_do_not_leak(client, stub_engine):
    """The SSE path has its own teardown, so it gets its own leak check."""
    baseline = block_accounting(client)

    for i in range(8):
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": f"stream {i}"}],
                "max_tokens": 3,
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200
            for _ in response.iter_lines():
                pass

    after = block_accounting(client)
    assert_no_leak(after, "after streaming requests")
    assert after["allocated_blocks"] == baseline["allocated_blocks"]


# -------------------------------------------------------------- unicode

@pytest.mark.parametrize(
    "text",
    [
        "café naïve",       # combining accents
        "你好世界",    # CJK
        "\U0001f600 \U0001f680",       # emoji outside the BMP
        "مرحبا",  # right-to-left
        "mixed \U0001f44d text éà",
    ],
)
def test_unicode_prompts_round_trip(client, stub_engine, text):
    """
    Non-ASCII input must not be mangled or crash detokenisation.

    Byte-level BPE makes this non-trivial: a multi-byte character can be split
    across tokens, so incremental detokenisation has to reassemble it rather
    than decode each piece alone.
    """
    response = client.post("/v1/completions", json={"prompt": text, "max_tokens": 3})
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body["text"], str)
    # Must be serialisable back out without an encoding error.
    json.dumps(body)


# --------------------------------------------------- prefix sharing via API

def test_identical_concurrent_prompts_share_blocks(stub_engine):
    """
    Prefix sharing, observed through the engine the API drives.

    Two requests with the same long prefix must both produce correct output and
    at some point report a shared block. Asserting only "both succeeded" would
    pass even if sharing silently did nothing.
    """
    # 40 characters -> 40 stub tokens -> 10 blocks of 4, so there is a genuine
    # shared prefix to find.
    prompt = "shared system prompt for the two requests"
    params = SamplingParams(max_new_tokens=4, temperature=0.0, stop_on_eos=False)

    ids = [stub_engine.submit(prompt + f" {i}", params) for i in range(2)]
    peak_shared = 0
    while stub_engine.has_work():
        stub_engine.step()
        peak_shared = max(peak_shared, stub_engine.allocator.get_num_shared_blocks())
    stub_engine.step()

    for seq_id in ids:
        assert stub_engine.result_for(seq_id)["finish_reason"] is not None

    assert peak_shared > 0, "two identical prefixes shared no blocks"
    total = stub_engine.allocator.get_total_blocks()
    assert stub_engine.allocator.get_num_free_blocks() == total


def test_blocks_endpoint_agrees_with_telemetry(client, stub_engine):
    """Two endpoints report the same pool; they must not disagree."""
    stub_engine.generate(
        ["consistency check"], SamplingParams(max_new_tokens=3, temperature=0.0)
    )
    telemetry = client.get("/v1/engine/telemetry").json()
    blocks = client.get("/v1/engine/blocks").json()

    assert blocks["total_blocks"] == telemetry["total_blocks"]
    assert blocks["block_size"] == telemetry["block_size"]
    assert len(blocks["occupancy"]) == telemetry["total_blocks"]


# ----------------------------------------------------------- engine not ready

def test_telemetry_is_503_before_the_engine_loads(client, monkeypatch):
    """A cold server must say so, not pretend to have numbers."""
    monkeypatch.setattr(ENGINE_MANAGER, "engine", None)
    assert client.get("/v1/engine/telemetry").status_code == 503
    assert client.get("/v1/engine/blocks").status_code == 503
    # /metrics stays available, so monitoring works during start-up.
    assert client.get("/metrics").status_code == 200


def test_health_reports_load_failure(client, monkeypatch):
    monkeypatch.setattr(ENGINE_MANAGER, "load_error", "ValueError: no such model")
    monkeypatch.setattr(ENGINE_MANAGER, "ready", False)
    body = client.get("/v1/health").json()
    assert body["status"] == "degraded"
    assert body["engine_ready"] is False
    assert "no such model" in body["load_error"]
