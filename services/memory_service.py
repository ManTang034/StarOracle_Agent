import hashlib
import json
from datetime import datetime

from langchain_chroma import Chroma
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from utils.config_handler import model_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from utils.prompt_loader import load_memory_prompt


class MemoryService:
    def __init__(self, chat_model: BaseChatModel):
        self.chat_model = chat_model
        self.embedding_model_name = model_conf["embedding"]["model_name"]
        self.collection_name = model_conf["vector_store"]["collection_name"]
        self.persist_directory = get_abs_path(model_conf["vector_store"]["persist_directory"])
        self.retrieval_limit = model_conf["vector_store"].get("retrieval_limit", 4)
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=DashScopeEmbeddings(model=self.embedding_model_name),
        )

    def retrieve_context(self, user_id: str, query: str, limit: int | None = None) -> str:
        effective_limit = limit or self.retrieval_limit
        try:
            documents = self.vector_store.similarity_search(
                query,
                k=effective_limit,
                filter={"user_id": user_id},
            )
        except Exception as exc:
            logger.error(f"Error retrieving memory for user_id={user_id}: {exc}")
            return ""

        if not documents:
            return ""

        return "\n".join(f"- {doc.page_content}" for doc in documents if doc.page_content)

    def remember(self, user_id: str, query: str, answer: str) -> None:
        memory_items = self._extract_memory_items(query, answer)
        if not memory_items:
            return

        timestamp = datetime.now().isoformat(timespec="seconds")
        ids = []
        documents = []
        metadatas = []

        for item in memory_items:
            normalized_item = item.strip()
            if not normalized_item:
                continue

            doc_id = hashlib.sha1(f"{user_id}:{normalized_item}".encode("utf-8")).hexdigest()
            ids.append(doc_id)
            documents.append(normalized_item)
            metadatas.append({"user_id": user_id, "source": "conversation", "created_at": timestamp})

        if not ids:
            return

        try:
            self.vector_store.add_texts(texts=documents, metadatas=metadatas, ids=ids)
        except Exception as exc:
            logger.error(f"Error saving memory for user_id={user_id}: {exc}")

    def _extract_memory_items(self, query: str, answer: str) -> list[str]:
        prompt = ChatPromptTemplate.from_template(load_memory_prompt())
        chain = prompt | self.chat_model | StrOutputParser()

        try:
            raw_result = chain.invoke({"query": query, "answer": answer})
        except Exception as exc:
            logger.error(f"Error extracting memory items: {exc}")
            return []

        return self._parse_memory_items(raw_result)

    def _parse_memory_items(self, raw_result: str) -> list[str]:
        text = raw_result.strip()
        if not text:
            return []

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass

        items = []
        for line in text.splitlines():
            normalized_line = line.strip().lstrip("-•*").strip()
            if normalized_line:
                items.append(normalized_line)
        return items