from __future__ import annotations

from tests.conftest import FakeChroma


def _memory_conf(tmp_path):
    return {
        "embedding": {"model_name": "dummy-embedding"},
        "vector_store": {"collection_name": "legacy_memory", "persist_directory": str(tmp_path / "memory"), "retrieval_limit": 4},
        "memory_store": {
            "collection_name": "test_memory",
            "persist_directory": str(tmp_path / "memory"),
            "retrieval_limit": 4,
            "preference_limit": 2,
            "fact_limit": 2,
            "recent_context_limit": 2,
            "preference_ttl_days": 365,
            "fact_ttl_days": 180,
            "recent_context_ttl_days": 7,
            "max_semantic_memories": 24,
            "max_recent_context_items": 6,
        },
    }


def test_memory_service_remember_and_retrieve_context(monkeypatch, tmp_path):
    import services.memory_service as memory_module

    monkeypatch.setattr(memory_module, "Chroma", FakeChroma)
    monkeypatch.setattr(memory_module, "DashScopeEmbeddings", lambda model: object())
    monkeypatch.setattr(memory_module, "get_abs_path", lambda path: str(tmp_path / path))
    monkeypatch.setattr(memory_module, "model_conf", _memory_conf(tmp_path))

    service = memory_module.MemoryService(chat_model=object())
    monkeypatch.setattr(
        service,
        "_extract_memory_items",
        lambda query, answer: [
            {"type": "preference", "content": "用户偏好简洁直接的回答"},
            {"type": "fact", "content": "用户是白羊座"},
        ],
    )

    saved_count = service.remember("demo-user", "我喜欢简洁一点", "好的")
    assert saved_count == 3

    context = service.retrieve_context("demo-user", "白羊座 简洁 直接")
    assert "【用户偏好】" in context
    assert "【用户事实】" in context
    assert "【最近对话上下文】" in context
    assert "用户偏好简洁直接的回答" in context
    assert "用户是白羊座" in context
    assert "我喜欢简洁一点" in context


def test_memory_cleanup_removes_expired_items(monkeypatch, tmp_path):
    import services.memory_service as memory_module

    monkeypatch.setattr(memory_module, "Chroma", FakeChroma)
    monkeypatch.setattr(memory_module, "DashScopeEmbeddings", lambda model: object())
    monkeypatch.setattr(memory_module, "get_abs_path", lambda path: str(tmp_path / path))
    monkeypatch.setattr(memory_module, "model_conf", _memory_conf(tmp_path))

    service = memory_module.MemoryService(chat_model=object())
    service.vector_store.add_texts(
        texts=["过期记忆", "有效记忆"],
        ids=["expired-id", "fresh-id"],
        metadatas=[
            {
                "id": "expired-id",
                "user_id": "demo-user",
                "memory_type": "fact",
                "created_at": "2000-01-01T00:00:00",
                "expires_at": "2000-01-02T00:00:00",
            },
            {
                "id": "fresh-id",
                "user_id": "demo-user",
                "memory_type": "fact",
                "created_at": "2099-01-01T00:00:00",
                "expires_at": "2099-12-31T00:00:00",
            },
        ],
    )

    service._cleanup_user_memories("demo-user")

    assert "expired-id" in service.vector_store.deleted_ids
    assert "fresh-id" not in service.vector_store.deleted_ids
