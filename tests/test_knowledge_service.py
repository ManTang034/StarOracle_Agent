from __future__ import annotations

from langchain_core.documents import Document

from tests.conftest import FakeChroma


def test_document_dedup_and_citation_formatting():
    from services.knowledge_service import (
        deduplicate_documents, format_retrieved_context_with_citations)

    documents = [
        Document(page_content="白羊座今天适合先行动。", metadata={"source_name": "note-a", "source_type": "text", "chunk_index": 0}),
        Document(page_content="白羊座今天适合先行动。", metadata={"source_name": "note-b", "source_type": "text", "chunk_index": 1}),
        Document(page_content="水瓶座今天适合先列清单。", metadata={"source_name": "note-c", "source_type": "text", "chunk_index": 2}),
    ]

    unique_documents = deduplicate_documents(documents)
    assert len(unique_documents) == 2

    cited_context = format_retrieved_context_with_citations(documents)
    assert "【1】白羊座今天适合先行动。" in cited_context
    assert "来源：note-a" in cited_context
    assert "来源：note-c" in cited_context


def test_knowledge_service_add_and_retrieve_context(monkeypatch, tmp_path):
    import services.knowledge_service as knowledge_module

    monkeypatch.setattr(knowledge_module, "Chroma", FakeChroma)
    monkeypatch.setattr(knowledge_module, "DashScopeEmbeddings", lambda model: object())
    monkeypatch.setattr(knowledge_module, "get_abs_path", lambda path: str(tmp_path / path))
    monkeypatch.setattr(
        knowledge_module,
        "model_conf",
        {
            "embedding": {"model_name": "dummy-embedding"},
            "knowledge_store": {
                "collection_name": "test_knowledge",
                "persist_directory": str(tmp_path / "knowledge"),
                "chunk_size": 32,
                "chunk_overlap": 0,
                "retrieval_limit": 3,
            },
        },
    )

    service = knowledge_module.KnowledgeService()
    result = service.add_texts("白羊座今天适合先行动，再复盘。", source_name="regression-note")

    assert result["status"] == "added"
    assert result["chunk_count"] >= 1

    context = service.retrieve_context("白羊座今天适合先行动吗？")
    assert "【1】" in context
    assert "白羊座今天适合先行动" in context
    assert "来源：regression-note" in context
