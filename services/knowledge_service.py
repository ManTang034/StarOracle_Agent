import hashlib
import os
from typing import Any

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config_handler import model_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class KnowledgeService:
    def __init__(self):
        # 知识库和记忆库都使用 Chroma 持久化，开源复现时本地会自动生成目录。
        self.embedding_model_name = model_conf["embedding"]["model_name"]
        self.collection_name = model_conf["knowledge_store"]["collection_name"]
        self.persist_directory = get_abs_path(model_conf["knowledge_store"]["persist_directory"])
        self.chunk_size = model_conf["knowledge_store"].get("chunk_size", 800)
        self.chunk_overlap = model_conf["knowledge_store"].get("chunk_overlap", 50)
        self.retrieval_limit = model_conf["knowledge_store"].get("retrieval_limit", 4)
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=DashScopeEmbeddings(model=self.embedding_model_name),
        )

    def add_urls(self, url: str) -> dict[str, Any]:
        # URL 入库前先做去重，避免重复抓取同一篇页面。
        normalized_url = url.strip()
        if not normalized_url:
            return self._ingest_result("url", url, "empty", 0, "URL 不能为空")

        if self._source_exists("url", normalized_url):
            return self._ingest_result("url", normalized_url, "exists", 0, f"URL 已存在数据库中: {normalized_url}")

        try:
            loader = WebBaseLoader(normalized_url)
            documents = loader.load()
            chunk_count = self._store_documents(documents, source_type="url", source_id=normalized_url, source_name=normalized_url)
            if chunk_count == 0:
                return self._ingest_result("url", normalized_url, "empty", 0, f"URL 未提取到可入库内容: {normalized_url}")
            return self._ingest_result("url", normalized_url, "added", chunk_count, f"已新增 {chunk_count} 条 URL 分块: {normalized_url}")
        except Exception as exc:
            logger.error(f"Error adding url knowledge for {normalized_url}: {exc}")
            return self._ingest_result("url", normalized_url, "error", 0, f"URL 入库失败: {normalized_url}")

    def add_pdfs(self, pdf_path: str) -> dict[str, Any]:
        # PDF 一般来自本地上传或本地路径输入，因此先做绝对路径标准化。
        normalized_pdf_path = os.path.abspath(pdf_path.strip())
        if not normalized_pdf_path or not os.path.exists(normalized_pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return self._ingest_result("pdf", normalized_pdf_path, "missing", 0, f"PDF 文件不存在: {pdf_path}")

        if self._source_exists("pdf", normalized_pdf_path):
            return self._ingest_result("pdf", normalized_pdf_path, "exists", 0, f"PDF 已存在数据库中: {normalized_pdf_path}")

        try:
            loader = PyPDFLoader(normalized_pdf_path)
            documents = loader.load()
            source_name = os.path.basename(normalized_pdf_path)
            chunk_count = self._store_documents(
                documents,
                source_type="pdf",
                source_id=normalized_pdf_path,
                source_name=source_name,
            )
            if chunk_count == 0:
                return self._ingest_result("pdf", normalized_pdf_path, "empty", 0, f"PDF 未提取到可入库内容: {normalized_pdf_path}")
            return self._ingest_result("pdf", normalized_pdf_path, "added", chunk_count, f"已新增 {chunk_count} 条 PDF 分块: {source_name}")
        except Exception as exc:
            logger.error(f"Error adding pdf knowledge for {normalized_pdf_path}: {exc}")
            return self._ingest_result("pdf", normalized_pdf_path, "error", 0, f"PDF 入库失败: {normalized_pdf_path}")

    def add_texts(self, text: str, source_name: str = "manual_text") -> dict[str, Any]:
        # 文本入库使用内容哈希做去重，避免同一段内容被反复写入。
        normalized_text = text.strip()
        if not normalized_text:
            return self._ingest_result("text", source_name, "empty", 0, "文本不能为空")

        text_hash = hashlib.sha1(normalized_text.encode("utf-8")).hexdigest()

        if self._source_exists("text", text_hash):
            return self._ingest_result("text", source_name, "exists", 0, f"文本已存在数据库中: {source_name}")

        try:
            documents = [
                Document(
                    page_content=normalized_text,
                    metadata={
                        "source_type": "text",
                        "source_id": text_hash,
                        "source_hash": text_hash,
                        "source_name": source_name,
                    },
                )
            ]
            chunk_count = self._store_documents(
                documents,
                source_type="text",
                source_id=text_hash,
                source_name=source_name,
            )
            if chunk_count == 0:
                return self._ingest_result("text", source_name, "empty", 0, f"文本未提取到可入库内容: {source_name}")
            return self._ingest_result("text", source_name, "added", chunk_count, f"已新增 {chunk_count} 条文本分块: {source_name}")
        except Exception as exc:
            logger.error(f"Error adding text knowledge for {source_name}: {exc}")
            return self._ingest_result("text", source_name, "error", 0, f"文本入库失败: {source_name}")

    def retrieve_context(self, query: str, limit: int | None = None) -> str:
        # RAG 检索返回拼接后的文本上下文，直接注入系统提示词。
        effective_limit = limit or self.retrieval_limit
        logger.info(f"Agent querying knowledge database via RAG: query={query!r}, limit={effective_limit}")
        try:
            documents = self.vector_store.similarity_search(query, k=effective_limit)
        except Exception as exc:
            logger.error(f"Error retrieving knowledge for query={query}: {exc}")
            return ""

        if not documents:
            return ""

        lines: list[str] = []
        for doc in documents:
            content = (doc.page_content or "").strip()
            if content:
                lines.append(content)

        return "\n".join(lines)

    def _source_exists(self, source_type: str, source_id: str) -> bool:
        try:
            result = self.vector_store.get(
                where={
                    "$and": [
                        {"source_type": source_type},
                        {"source_id": source_id},
                    ]
                },
                limit=1,
            )
            return bool(result.get("ids"))
        except Exception as exc:
            logger.warning(f"Error checking existing knowledge for {source_type}:{source_id}: {exc}")
            return False

    def _ingest_result(self, source_type: str, source_name: str, status: str, chunk_count: int, message: str) -> dict[str, Any]:
        return {
            "status": status,
            "source_type": source_type,
            "source_name": source_name,
            "chunk_count": chunk_count,
            "message": message,
        }

    def _store_documents(self, documents, source_type: str, source_id: str, source_name: str) -> int:
        # 先切分再写入，确保长文档在检索时更容易命中相关片段。
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = splitter.split_documents(documents)
        if not chunks:
            return 0

        texts: list[str] = []
        metadatas: list[dict] = []
        ids: list[str] = []

        for index, chunk in enumerate(chunks):
            page_content = chunk.page_content.strip()
            if not page_content:
                continue

            metadata = dict(chunk.metadata or {})
            metadata.update({
                "source_type": source_type,
                "source_id": source_id,
                "source_name": source_name,
                "chunk_index": index,
            })

            doc_id = hashlib.sha1(f"{source_type}:{source_id}:{index}:{page_content}".encode("utf-8")).hexdigest()
            texts.append(page_content)
            metadatas.append(metadata)
            ids.append(doc_id)

        if not texts:
            return 0

        self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return len(texts)
