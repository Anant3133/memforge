import pytest
from fastapi.testclient import TestClient
from server.app import app


def test_health_check_endpoint():
    client = TestClient(app)
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "MemForge" in data["service"]


def test_models_endpoint():
    client = TestClient(app)
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) >= 1


def test_chat_completions_endpoint():
    client = TestClient(app)
    payload = {
        "messages": [
            {"role": "user", "content": "What is PagedAttention?"}
        ],
        "max_new_tokens": 16,
        "temperature": 0.7,
        "use_speculative": False
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "content" in data["choices"][0]["message"]
