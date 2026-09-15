from __future__ import annotations

from fastapi.testclient import TestClient


def test_core_chat_and_ingest_endpoints(monkeypatch):
    import server

    class DummyMaster:
        def run(self, query: str, user_id: str = "default"):
            return f"reply:{user_id}:{query}"

    class DummyKnowledgeService:
        def add_texts(self, text: str, source_name: str = "manual_text"):
            return {"status": "added", "source_type": "text", "source_name": source_name, "chunk_count": 1, "message": text}

    monkeypatch.setattr(server, "Master", DummyMaster)
    monkeypatch.setattr(server, "KnowledgeService", DummyKnowledgeService)

    client = TestClient(server.app)

    chat_response = client.post("/chat", params={"query": "你好", "user_id": "demo-user"}, headers={"X-TRACE-ID": "trace-123"})
    assert chat_response.status_code == 200
    assert chat_response.json()["message"] == "reply:demo-user:你好"
    assert chat_response.json()["trace_id"] == "trace-123"

    text_response = client.post(
        "/add_texts",
        params={"source_name": "regression-text"},
        data="白羊座今天适合先行动。",
        headers={"content-type": "text/plain", "X-TRACE-ID": "trace-456"},
    )
    assert text_response.status_code == 200
    assert text_response.json()["status"] == "added"
    assert text_response.json()["source_name"] == "regression-text"
    assert text_response.json()["trace_id"] == "trace-456"
