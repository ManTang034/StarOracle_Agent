import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta

from langchain_chroma import Chroma
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from utils.config_handler import model_conf
from utils.logger_handler import logger
from utils.observability import is_trace_debug
from utils.path_tool import get_abs_path
from utils.prompt_loader import load_memory_prompt

MEMORY_TYPE_PREFERENCE = "preference"
MEMORY_TYPE_FACT = "fact"
MEMORY_TYPE_RECENT_CONTEXT = "recent_context"
MEMORY_TYPES = {MEMORY_TYPE_PREFERENCE, MEMORY_TYPE_FACT, MEMORY_TYPE_RECENT_CONTEXT}


class MemoryService:
    def __init__(self, chat_model: BaseChatModel):
        # 长期记忆继续走向量库，但在 metadata 上拆成偏好、事实和最近上下文三层。
        self.chat_model = chat_model
        self.embedding_model_name = model_conf["embedding"]["model_name"]
        memory_conf = model_conf.get("memory_store", {})
        vector_conf = model_conf["vector_store"]
        self.collection_name = memory_conf.get("collection_name", vector_conf["collection_name"])
        self.persist_directory = get_abs_path(memory_conf.get("persist_directory", vector_conf["persist_directory"]))
        self.retrieval_limit = memory_conf.get("retrieval_limit", vector_conf.get("retrieval_limit", 4))
        self.preference_limit = memory_conf.get("preference_limit", 2)
        self.fact_limit = memory_conf.get("fact_limit", 2)
        self.recent_context_limit = memory_conf.get("recent_context_limit", 3)
        self.preference_ttl_days = memory_conf.get("preference_ttl_days", 365)
        self.fact_ttl_days = memory_conf.get("fact_ttl_days", 180)
        self.recent_context_ttl_days = memory_conf.get("recent_context_ttl_days", 7)
        self.max_semantic_memories = memory_conf.get("max_semantic_memories", 24)
        self.max_recent_context_items = memory_conf.get("max_recent_context_items", 6)
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=DashScopeEmbeddings(model=self.embedding_model_name),
        )

    def retrieve_context(self, user_id: str, query: str, limit: int | None = None) -> str:
        # 先清理过期和过量的记忆，再按“偏好 / 事实 / 最近上下文”分别召回。
        self._cleanup_user_memories(user_id)

        effective_limit = limit or self.retrieval_limit
        try:
            preference_documents = self._retrieve_semantic_memories(
                user_id=user_id,
                query=query,
                memory_type=MEMORY_TYPE_PREFERENCE,
                limit=self.preference_limit,
                fallback_limit=effective_limit,
            )
            fact_documents = self._retrieve_semantic_memories(
                user_id=user_id,
                query=query,
                memory_type=MEMORY_TYPE_FACT,
                limit=self.fact_limit,
                fallback_limit=effective_limit,
            )
            recent_context_documents = self._retrieve_recent_context(user_id, self.recent_context_limit)
        except Exception as exc:
            logger.error(f"Error retrieving memory for user_id={user_id}: {exc}")
            return ""

        sections = []
        if preference_documents:
            sections.append(self._format_memory_section("用户偏好", preference_documents))
        if fact_documents:
            sections.append(self._format_memory_section("用户事实", fact_documents))
        if recent_context_documents:
            sections.append(self._format_memory_section("最近对话上下文", recent_context_documents))

        if not sections:
            return ""

        return "\n\n".join(sections)

    def remember(self, user_id: str, query: str, answer: str) -> int:
        # 从一轮问答里提取可长期保存的偏好、事实，并额外保存最近一轮对话上下文。
        memory_items = self._extract_memory_items(query, answer)
        if is_trace_debug():
            logger.info(f"[memory] extracted_items={memory_items}")
        else:
            logger.info(f"[memory] extracted_count={len(memory_items)}")

        timestamp = datetime.now()
        timestamp_text = timestamp.isoformat(timespec="seconds")
        ids = []
        documents = []
        metadatas = []

        for item in memory_items:
            memory_type = self._normalize_memory_type(item.get("type"))
            if memory_type not in {MEMORY_TYPE_PREFERENCE, MEMORY_TYPE_FACT}:
                continue

            normalized_item = item.get("content", "").strip()
            if not normalized_item:
                continue

            doc_id = hashlib.sha1(f"{user_id}:{memory_type}:{normalized_item}".encode("utf-8")).hexdigest()
            if self._document_exists(doc_id):
                continue

            ids.append(doc_id)
            documents.append(normalized_item)
            metadatas.append(
                self._build_memory_metadata(
                    user_id=user_id,
                    memory_type=memory_type,
                    source="conversation",
                    created_at=timestamp_text,
                    content_hash=self._content_hash(normalized_item),
                )
            )

        recent_context = self._build_recent_context_record(user_id, query, answer, timestamp)
        if recent_context and not self._document_exists(recent_context["id"]):
            ids.append(recent_context["id"])
            documents.append(recent_context["content"])
            metadatas.append(recent_context["metadata"])

        if not ids:
            self._cleanup_user_memories(user_id)
            return 0

        try:
            self.vector_store.add_texts(texts=documents, metadatas=metadatas, ids=ids)
            logger.info(f"[memory] saved_count={len(ids)}")
            self._cleanup_user_memories(user_id)
        except Exception as exc:
            logger.error(f"Error saving memory for user_id={user_id}: {exc}")
            return 0

        return len(ids)

    def _extract_memory_items(self, query: str, answer: str) -> list[dict[str, str]]:
        # 使用专门的记忆抽取提示词，让模型输出更适合结构化存储的结果。
        prompt = ChatPromptTemplate.from_template(load_memory_prompt())
        chain = prompt | self.chat_model | StrOutputParser()

        try:
            raw_result = chain.invoke({"query": query, "answer": answer})
        except Exception as exc:
            logger.error(f"Error extracting memory items: {exc}")
            return []

        return self._parse_memory_items(raw_result)

    def _parse_memory_items(self, raw_result: str) -> list[dict[str, str]]:
        text = raw_result.strip()
        if not text:
            return []

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                items: list[dict[str, str]] = []
                for item in parsed:
                    if isinstance(item, dict):
                        content = str(item.get("content", "")).strip()
                        if not content:
                            continue
                        items.append({"type": self._normalize_memory_type(str(item.get("type", "fact"))), "content": content})
                        continue

                    content = str(item).strip()
                    if content:
                        items.append({"type": MEMORY_TYPE_FACT, "content": content})
                return items
        except json.JSONDecodeError:
            pass

        items: list[dict[str, str]] = []
        for line in text.splitlines():
            normalized_line = line.strip().lstrip("-•*").strip()
            if normalized_line:
                items.append({"type": MEMORY_TYPE_FACT, "content": normalized_line})
        return items

    def _normalize_memory_type(self, memory_type: str | None) -> str:
        normalized = (memory_type or MEMORY_TYPE_FACT).strip().lower()
        return normalized if normalized in MEMORY_TYPES else MEMORY_TYPE_FACT

    def _content_hash(self, content: str) -> str:
        normalized_content = " ".join(content.split()).lower()
        return hashlib.sha1(normalized_content.encode("utf-8")).hexdigest()

    def _truncate_text(self, text: str, max_length: int = 240) -> str:
        normalized_text = text.strip()
        if len(normalized_text) <= max_length:
            return normalized_text
        return f"{normalized_text[:max_length].rstrip()}..."

    def _build_memory_metadata(
        self,
        *,
        user_id: str,
        memory_type: str,
        source: str,
        created_at: str,
        content_hash: str,
    ) -> dict[str, str]:
        ttl_days = self._memory_ttl_days(memory_type)
        expires_at = (datetime.fromisoformat(created_at) + timedelta(days=ttl_days)).isoformat(timespec="seconds")
        return {
            "user_id": user_id,
            "memory_type": memory_type,
            "source": source,
            "created_at": created_at,
            "expires_at": expires_at,
            "content_hash": content_hash,
        }

    def _memory_ttl_days(self, memory_type: str) -> int:
        if memory_type == MEMORY_TYPE_PREFERENCE:
            return self.preference_ttl_days
        if memory_type == MEMORY_TYPE_RECENT_CONTEXT:
            return self.recent_context_ttl_days
        return self.fact_ttl_days

    def _build_recent_context_record(self, user_id: str, query: str, answer: str, timestamp: datetime) -> dict[str, object] | None:
        query_text = self._truncate_text(query, 320)
        answer_text = self._truncate_text(answer, 480)
        if not query_text and not answer_text:
            return None

        content = f"用户：{query_text}\n助手：{answer_text}"
        content_hash = self._content_hash(content)
        created_at = timestamp.isoformat(timespec="seconds")
        doc_id = hashlib.sha1(
            f"{user_id}:{MEMORY_TYPE_RECENT_CONTEXT}:{created_at}:{content_hash}".encode("utf-8")
        ).hexdigest()
        return {
            "id": doc_id,
            "content": content,
            "metadata": self._build_memory_metadata(
                user_id=user_id,
                memory_type=MEMORY_TYPE_RECENT_CONTEXT,
                source="conversation_turn",
                created_at=created_at,
                content_hash=content_hash,
            ),
        }

    def _document_exists(self, doc_id: str) -> bool:
        try:
            result = self.vector_store.get(ids=[doc_id], limit=1)
            return bool(result.get("ids"))
        except Exception:
            return False

    def _cleanup_user_memories(self, user_id: str) -> None:
        try:
            result = self.vector_store.get(where={"user_id": user_id})
        except Exception as exc:
            logger.warning(f"Error loading memory for cleanup user_id={user_id}: {exc}")
            return

        ids = result.get("ids", [])
        metadatas = result.get("metadatas", [])
        if not ids or not metadatas:
            return

        now = datetime.now()
        ids_to_delete: list[str] = []
        recent_context_candidates: list[tuple[datetime, str]] = []
        semantic_candidates: dict[str, list[tuple[datetime, str]]] = defaultdict(list)

        for doc_id, metadata in zip(ids, metadatas):
            if not isinstance(metadata, dict):
                continue

            expires_at = self._parse_datetime(metadata.get("expires_at"))
            if expires_at and expires_at <= now:
                ids_to_delete.append(doc_id)
                continue

            created_at = self._parse_datetime(metadata.get("created_at")) or now
            memory_type = metadata.get("memory_type")
            if memory_type == MEMORY_TYPE_RECENT_CONTEXT:
                recent_context_candidates.append((created_at, doc_id))
            elif memory_type in {MEMORY_TYPE_PREFERENCE, MEMORY_TYPE_FACT}:
                semantic_candidates[memory_type].append((created_at, doc_id))

        recent_context_candidates.sort(key=lambda item: item[0], reverse=True)
        ids_to_delete.extend(doc_id for _, doc_id in recent_context_candidates[self.max_recent_context_items :])

        for memory_type, candidates in semantic_candidates.items():
            candidates.sort(key=lambda item: item[0], reverse=True)
            ids_to_delete.extend(doc_id for _, doc_id in candidates[self.max_semantic_memories :])

        if ids_to_delete:
            unique_ids = list(dict.fromkeys(ids_to_delete))
            try:
                self.vector_store.delete(ids=unique_ids)
                logger.info(f"[memory] cleanup_deleted_count={len(unique_ids)} user_id={user_id}")
            except Exception as exc:
                logger.warning(f"Error deleting expired memory for user_id={user_id}: {exc}")

    def _retrieve_semantic_memories(
        self,
        *,
        user_id: str,
        query: str,
        memory_type: str,
        limit: int,
        fallback_limit: int,
    ) -> list:
        # 先尝试带类型过滤的召回；如果旧数据还没有 memory_type，则回退到用户级召回。
        raw_limit = max(limit * 2, limit + 2)
        typed_filter = {"$and": [{"user_id": user_id}, {"memory_type": memory_type}]}
        fallback_filter = {"user_id": user_id}

        typed_documents = self.vector_store.similarity_search(query, k=raw_limit, filter=typed_filter)
        if typed_documents:
            return self._deduplicate_documents(typed_documents)[:limit]

        fallback_documents = self.vector_store.similarity_search(query, k=max(raw_limit, fallback_limit), filter=fallback_filter)
        fallback_documents = [doc for doc in fallback_documents if doc.metadata.get("memory_type") in {memory_type, None, ""}]
        return self._deduplicate_documents(fallback_documents)[:limit]

    def _retrieve_recent_context(self, user_id: str, limit: int) -> list:
        try:
            result = self.vector_store.get(
                where={
                    "$and": [
                        {"user_id": user_id},
                        {"memory_type": MEMORY_TYPE_RECENT_CONTEXT},
                    ]
                }
            )
        except Exception as exc:
            logger.warning(f"Error loading recent context for user_id={user_id}: {exc}")
            return []

        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        candidates = []

        for doc_id, content, metadata in zip(ids, documents, metadatas):
            if not content:
                continue
            created_at = self._parse_datetime(metadata.get("created_at")) if isinstance(metadata, dict) else None
            candidates.append((created_at or datetime.min, doc_id, content, metadata or {}))

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = candidates[:limit]
        return [self._document_from_tuple(item) for item in selected]

    def _document_from_tuple(self, item: tuple) -> object:
        from langchain_core.documents import Document

        _, _, content, metadata = item
        return Document(page_content=content, metadata=metadata)

    def _format_memory_section(self, title: str, documents) -> str:
        lines = [f"【{title}】"]
        for doc in documents:
            if not getattr(doc, "page_content", ""):
                continue
            lines.append(f"- {doc.page_content.strip()}")
        return "\n".join(lines)

    def _deduplicate_documents(self, documents) -> list:
        seen_hashes: set[str] = set()
        unique_documents = []
        for doc in documents:
            content = getattr(doc, "page_content", "")
            if not content:
                continue
            content_hash = self._content_hash(content)
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            unique_documents.append(doc)
        return unique_documents

    def _parse_datetime(self, value) -> datetime | None:
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        text = str(value).strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None