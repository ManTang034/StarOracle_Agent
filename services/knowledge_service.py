import hashlib
import os
import re
from typing import Any

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config_handler import model_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


def _normalize_text(value: str) -> str:
    # 统一空白和大小写，确保同一段内容在去重和哈希时得到稳定结果。
    return re.sub(r"\s+", " ", value or "").strip().lower()


def _document_content_hash(document: Document) -> str:
    # 只基于正文内容计算哈希，避免 metadata 变化导致同一内容被视为不同文档。
    return hashlib.sha1(_normalize_text(document.page_content).encode("utf-8")).hexdigest()


def deduplicate_documents(documents: list[Document]) -> list[Document]:
    # 检索结果可能因为模板、页眉页脚或切块重叠而重复，这里按正文内容做去重。
    seen_hashes: set[str] = set()
    unique_documents: list[Document] = []

    for document in documents:
        content = (document.page_content or "").strip()
        if not content:
            continue

        content_hash = _document_content_hash(document)
        if content_hash in seen_hashes:
            continue

        seen_hashes.add(content_hash)
        unique_documents.append(document)

    return unique_documents


def format_cited_context(documents: list[Document]) -> str:
    # 把检索结果整理成“正文 + 来源信息”的格式，方便系统提示词直接消费并让模型引用来源。
    lines: list[str] = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        content = (document.page_content or "").strip()
        if not content:
            continue

        source_name = metadata.get("source_name") or metadata.get("source_id") or "unknown"
        source_type = metadata.get("source_type") or "unknown"
        chunk_index = metadata.get("chunk_index")
        citation_parts = [f"来源：{source_name}", f"类型：{source_type}"]
        if chunk_index is not None:
            citation_parts.append(f"分块：{chunk_index}")

        lines.append(f"【{index}】{content}")
        lines.append("  " + " | ".join(citation_parts))

    return "\n".join(lines)


def format_retrieved_context_with_citations(documents: list[Document]) -> str:
    return format_cited_context(deduplicate_documents(documents))


class KnowledgeService:
    def __init__(self):
        # 知识库和记忆库都使用 Chroma 持久化，开源复现时本地会自动生成目录。
        # 这里保留独立的知识库配置，后续如果要调检索质量，只需要改 config 里的参数。
        self.embedding_model_name = model_conf["embedding"]["model_name"]
        self.collection_name = model_conf["knowledge_store"]["collection_name"]
        self.persist_directory = get_abs_path(model_conf["knowledge_store"]["persist_directory"])
        # 较小的 chunk 让单个分块更聚焦，较大的 overlap 则帮助保留上下文连续性。
        self.chunk_size = model_conf["knowledge_store"].get("chunk_size", 600)
        self.chunk_overlap = model_conf["knowledge_store"].get("chunk_overlap", 80)
        # 召回时默认取前几个结果，避免上下文过长导致提示词被噪声淹没。
        self.retrieval_limit = model_conf["knowledge_store"].get("retrieval_limit", 4)
        # 中文文本更依赖标点和换行来保持语义完整，因此这里显式指定切分边界。
        self.chunk_separators = model_conf["knowledge_store"].get(
            "chunk_separators",
            ["\n\n", "\n", "。", "！", "？", "；", "，", " "]
        )
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=DashScopeEmbeddings(model=self.embedding_model_name),
        )

    def add_urls(self, url: str) -> dict[str, Any]:
        # URL 入库前先做去重，避免重复抓取同一篇页面。
        # 这里把 URL 作为 source_id，确保同一个页面不会重复抓取并写入向量库。
        normalized_url = url.strip()
        if not normalized_url:
            return self._ingest_result("url", url, "empty", 0, "URL 不能为空")

        if self._source_exists("url", normalized_url):
            return self._ingest_result("url", normalized_url, "exists", 0, f"URL 已存在数据库中: {normalized_url}")

        try:
            # WebBaseLoader 负责抓取网页正文，再交给统一的切分与写入逻辑处理。
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
        # 把路径标准化后，再把它当作 source_id，便于判断同一份 PDF 是否已经入库。
        normalized_pdf_path = os.path.abspath(pdf_path.strip())
        if not normalized_pdf_path or not os.path.exists(normalized_pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return self._ingest_result("pdf", normalized_pdf_path, "missing", 0, f"PDF 文件不存在: {pdf_path}")

        if self._source_exists("pdf", normalized_pdf_path):
            return self._ingest_result("pdf", normalized_pdf_path, "exists", 0, f"PDF 已存在数据库中: {normalized_pdf_path}")

        try:
            # PyPDFLoader 会把 PDF 按页解析成 Document 列表，后续再统一切分。
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
        # 这种方式适合“复制一段资料再入库”的场景，比用文件名更稳。
        normalized_text = text.strip()
        if not normalized_text:
            return self._ingest_result("text", source_name, "empty", 0, "文本不能为空")

        text_hash = hashlib.sha1(normalized_text.encode("utf-8")).hexdigest()

        if self._source_exists("text", text_hash):
            return self._ingest_result("text", source_name, "exists", 0, f"文本已存在数据库中: {source_name}")

        try:
            # 手工输入的纯文本不需要额外解析，直接封装成 Document 进入统一管线。
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
        # 这里返回的是“可直接喂给模型的上下文”，而不是原始检索对象。
        effective_limit = limit or self.retrieval_limit
        logger.info(f"Agent querying knowledge database via RAG: query={query!r}, limit={effective_limit}")
        try:
            documents = self.retrieve_documents(query, limit=effective_limit)
        except Exception as exc:
            logger.error(f"Error retrieving knowledge for query={query}: {exc}")
            return ""

        if not documents:
            return ""

        return format_retrieved_context_with_citations(documents)

    def retrieve_documents(self, query: str, limit: int | None = None) -> list[Document]:
        # 先多取一些候选，再做去重和截断，能降低前几个结果刚好都是重复内容的概率。
        effective_limit = limit or self.retrieval_limit
        raw_limit = max(effective_limit * 2, effective_limit + 2)
        documents = self.vector_store.similarity_search(query, k=raw_limit)
        return deduplicate_documents(documents)[:effective_limit]

    def _source_exists(self, source_type: str, source_id: str) -> bool:
        # 先按来源级别做一次快速检查，避免重复入库同一 URL / PDF / 文本。
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
        # 统一返回入库结果结构，前端可以直接展示状态、分块数和消息。
        return {
            "status": status,
            "source_type": source_type,
            "source_name": source_name,
            "chunk_count": chunk_count,
            "message": message,
        }

    def _store_documents(self, documents, source_type: str, source_id: str, source_name: str) -> int:
        # 先切分再写入，确保长文档在检索时更容易命中相关片段。
        # 这里是知识库入库的核心管线：切分、去重、补 metadata、生成稳定 id、批量写入。
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
        seen_content_hashes: set[str] = set()

        for index, chunk in enumerate(chunks):
            page_content = chunk.page_content.strip()
            if not page_content:
                continue

            # 同一批 chunk 内如果文本完全一致，直接跳过，避免冗余数据污染召回结果。
            content_hash = hashlib.sha1(_normalize_text(page_content).encode("utf-8")).hexdigest()
            if content_hash in seen_content_hashes:
                continue
            seen_content_hashes.add(content_hash)

            # metadata 既用于后续引用来源，也用于调试检索命中了哪一块内容。
            metadata = dict(chunk.metadata or {})
            metadata.update({
                "source_type": source_type,
                "source_id": source_id,
                "source_name": source_name,
                "chunk_index": index,
                "content_hash": content_hash,
            })

            # 用来源 + 分块序号 + 内容哈希生成稳定 id，既能避免冲突，也方便重复写入时保持可预测性。
            doc_id = hashlib.sha1(f"{source_type}:{source_id}:{index}:{content_hash}".encode("utf-8")).hexdigest()
            texts.append(page_content)
            metadatas.append(metadata)
            ids.append(doc_id)

        if not texts:
            return 0

        self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return len(texts)
