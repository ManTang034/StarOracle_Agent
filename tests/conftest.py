from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.documents import Document


def _matches_filter(metadata: dict, where: dict | None) -> bool:
    if not where:
        return True

    conditions = where.get("$and") if isinstance(where, dict) else None
    if conditions:
        return all(_matches_filter(metadata, condition) for condition in conditions)

    for key, expected_value in where.items():
        if metadata.get(key) != expected_value:
            return False
    return True


@dataclass
class FakeChroma:
    collection_name: str | None = None
    persist_directory: str | None = None
    embedding_function: object | None = None
    documents: list[Document] = field(default_factory=list)
    deleted_ids: list[str] = field(default_factory=list)

    def add_texts(self, texts, metadatas=None, ids=None):
        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [None] * len(texts)
        for text, metadata, doc_id in zip(texts, metadatas, ids):
            stored_metadata = dict(metadata or {})
            if doc_id is not None:
                stored_metadata.setdefault("id", doc_id)
            self.documents.append(Document(page_content=text, metadata=stored_metadata))

    def get(self, ids=None, where=None, limit=None):
        matched_documents = []
        for document in self.documents:
            metadata = document.metadata or {}
            if ids is not None and metadata.get("id") not in set(ids):
                continue
            if where is not None and not _matches_filter(metadata, where):
                continue
            matched_documents.append(document)

        if limit is not None:
            matched_documents = matched_documents[:limit]

        return {
            "ids": [document.metadata.get("id") for document in matched_documents],
            "documents": [document.page_content for document in matched_documents],
            "metadatas": [document.metadata for document in matched_documents],
        }

    def similarity_search(self, query, k=4, filter=None):
        normalized_query = re.sub(r"\s+", "", str(query).lower())
        query_tokens = set()
        if normalized_query:
            if len(normalized_query) <= 2:
                query_tokens.add(normalized_query)
            else:
                query_tokens.update(normalized_query[index : index + 2] for index in range(len(normalized_query) - 1))
        ranked = []

        for document in self.documents:
            metadata = document.metadata or {}
            if filter is not None and not _matches_filter(metadata, filter):
                continue

            content = (document.page_content or "").lower()
            score = sum(1 for token in query_tokens if token and token in content)
            ranked.append((score, document))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [document for score, document in ranked[:k] if score > 0 or not query_tokens]

    def delete(self, ids):
        ids_set = set(ids)
        self.deleted_ids.extend(ids)
        self.documents = [document for document in self.documents if document.metadata.get("id") not in ids_set]
