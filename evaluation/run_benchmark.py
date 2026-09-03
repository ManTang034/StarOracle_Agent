from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from tools.agent_tools import current_time


class LocalHashEmbeddings(Embeddings):
    """Deterministic local embeddings for offline benchmark stores."""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", "", text.lower().strip())

    def _vectorize(self, text: str) -> list[float]:
        normalized = self._normalize(text)
        vector = [0.0] * self.dimension
        if not normalized:
            return vector

        grams: list[str] = [normalized[index : index + 2] for index in range(max(len(normalized) - 1, 1))]
        if not grams:
            grams = [normalized]

        for gram in grams:
            digest = hashlib.sha1(gram.encode("utf-8")).hexdigest()
            slot = int(digest, 16) % self.dimension
            vector[slot] += 1.0

        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(text)


def load_benchmark(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def create_store(collection_name: str, persist_directory: str) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=LocalHashEmbeddings(),
    )


def seed_knowledge_store(store: Chroma, docs: list[dict[str, Any]]) -> None:
    if not docs:
        return

    texts = []
    ids = []
    metadatas = []
    for doc in docs:
        texts.append(doc["text"])
        ids.append(doc["id"])
        metadatas.append({"source_type": "benchmark_knowledge", "source_id": doc["id"], "source_name": doc["id"]})

    store.add_texts(texts=texts, ids=ids, metadatas=metadatas)


def seed_memory_store(store: Chroma, docs: list[dict[str, Any]]) -> None:
    if not docs:
        return

    texts = []
    ids = []
    metadatas = []
    for doc in docs:
        texts.append(doc["text"])
        ids.append(doc["id"])
        metadatas.append({"user_id": doc["user_id"], "source": "benchmark_memory", "source_id": doc["id"], "source_name": doc["id"]})

    store.add_texts(texts=texts, ids=ids, metadatas=metadatas)


def keyword_score(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    hits = sum(1 for keyword in keywords if keyword and keyword in text)
    return hits / len(keywords)


def retrieval_hit(documents: list[Document], expected_refs: list[str]) -> bool:
    if not expected_refs:
        return False
    expected = set(expected_refs)
    for document in documents:
        source_id = str((document.metadata or {}).get("source_id", ""))
        if source_id in expected:
            return True
    return False


def format_duration(ms: float) -> str:
    return f"{ms:.1f} ms"


def run_retrieval_benchmark(knowledge_store: Chroma, memory_store: Chroma, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in cases:
        category = case["category"]
        if category not in {"knowledge", "memory"}:
            continue

        store = knowledge_store if category == "knowledge" else memory_store
        query = case["query"]
        top_k = int(case.get("top_k", 3))
        user_filter = case.get("user_id")

        start = time.perf_counter()
        if category == "memory" and user_filter:
            retrieved = store.similarity_search(query, k=top_k, filter={"user_id": user_filter})
        else:
            retrieved = store.similarity_search(query, k=top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000

        hit = retrieval_hit(retrieved, case.get("knowledge_refs", []) if category == "knowledge" else case.get("memory_refs", []))
        result_text = "\n".join(document.page_content for document in retrieved if document.page_content)

        results.append(
            {
                "id": case["id"],
                "category": category,
                "retrieval_hit": hit,
                "retrieval_score": 1.0 if hit else 0.0,
                "latency_ms": elapsed_ms,
                "retrieved_text": result_text,
                "retrieved_ids": [(document.metadata or {}).get("source_id") for document in retrieved],
            }
        )

    return results


def run_direct_tool_benchmark(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in cases:
        if case["category"] != "tool":
            continue

        tool_name = case["tool_name"]
        start = time.perf_counter()
        if tool_name == "current_time":
            output = current_time.invoke({})
        else:
            output = f"unsupported tool: {tool_name}"
        elapsed_ms = (time.perf_counter() - start) * 1000

        expected_regex = case.get("expected_regex", "")
        passed = bool(re.search(expected_regex, str(output))) if expected_regex else bool(output)

        results.append(
            {
                "id": case["id"],
                "category": "tool",
                "passed": passed,
                "latency_ms": elapsed_ms,
                "output": str(output),
            }
        )

    return results


def run_live_agent_benchmark(
    knowledge_store: Chroma,
    memory_store: Chroma,
    cases: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    if not os.environ.get("DASHSCOPE_API_KEY"):
        return [], "DASHSCOPE_API_KEY not set; skipped live agent benchmark."

    from services.chat_service import Master

    results: list[dict[str, Any]] = []
    for case in cases:
        if case["category"] not in {"knowledge", "memory"}:
            continue

        master = Master()
        master.knowledge_service.vector_store = knowledge_store
        master.memory_service.vector_store = memory_store
        master.memory_service.remember = lambda *args, **kwargs: None

        user_id = case.get("user_id", "demo-user")
        query = case["query"]
        expected_keywords = case.get("expected_answer_keywords", [])

        start = time.perf_counter()
        answer = master.run(query, user_id=user_id)
        elapsed_ms = (time.perf_counter() - start) * 1000

        results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "answer_keyword_score": keyword_score(answer, expected_keywords),
                "latency_ms": elapsed_ms,
                "answer": answer,
            }
        )

    return results, None


def summarize_results(retrieval_results: list[dict[str, Any]], tool_results: list[dict[str, Any]], live_results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    if retrieval_results:
        summary["retrieval_cases"] = len(retrieval_results)
        summary["retrieval_hit_rate"] = sum(1 for result in retrieval_results if result["retrieval_hit"]) / len(retrieval_results)
        summary["retrieval_avg_latency_ms"] = sum(result["latency_ms"] for result in retrieval_results) / len(retrieval_results)

    if tool_results:
        summary["tool_cases"] = len(tool_results)
        summary["tool_pass_rate"] = sum(1 for result in tool_results if result["passed"]) / len(tool_results)
        summary["tool_avg_latency_ms"] = sum(result["latency_ms"] for result in tool_results) / len(tool_results)

    if live_results:
        summary["live_cases"] = len(live_results)
        summary["live_keyword_coverage"] = sum(result["answer_keyword_score"] for result in live_results) / len(live_results)
        summary["live_avg_latency_ms"] = sum(result["latency_ms"] for result in live_results) / len(live_results)

    return summary


def build_markdown_report(summary: dict[str, Any], retrieval_results: list[dict[str, Any]], tool_results: list[dict[str, Any]], live_results: list[dict[str, Any]]) -> str:
    lines = ["# Benchmark Report", ""]

    if summary:
        lines.append("## Summary")
        for key, value in summary.items():
            if isinstance(value, float):
                if "rate" in key or "coverage" in key:
                    display = f"{value:.2%}"
                else:
                    display = format_duration(value)
            else:
                display = str(value)
            lines.append(f"- {key}: {display}")
        lines.append("")

    if retrieval_results:
        lines.append("## Retrieval")
        for result in retrieval_results:
            lines.append(
                f"- {result['id']}: hit={result['retrieval_hit']} latency={format_duration(result['latency_ms'])} ids={result['retrieved_ids']}"
            )
        lines.append("")

    if tool_results:
        lines.append("## Tools")
        for result in tool_results:
            lines.append(f"- {result['id']}: passed={result['passed']} latency={format_duration(result['latency_ms'])}")
        lines.append("")

    if live_results:
        lines.append("## Live Agent")
        for result in live_results:
            lines.append(
                f"- {result['id']}: score={result['answer_keyword_score']:.2f} latency={format_duration(result['latency_ms'])} answer={result['answer']}"
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the StarOracle Agent benchmark.")
    parser.add_argument("--benchmark-file", default=str(Path(__file__).with_name("benchmark_cases.json")), help="Path to benchmark definition JSON.")
    parser.add_argument("--output", default="eval_report/report.json", help="Optional path to write the JSON report.")
    parser.add_argument("--markdown-output", default="", help="Optional path to write the markdown report.")
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark_file)
    benchmark = load_benchmark(benchmark_path)
    cases = benchmark.get("cases", [])

    temp_root = Path(tempfile.mkdtemp(prefix="staroracle_benchmark_"))
    try:
        knowledge_store = create_store("benchmark_knowledge", str(temp_root / "knowledge"))
        memory_store = create_store("benchmark_memory", str(temp_root / "memory"))

        seed_knowledge_store(knowledge_store, benchmark.get("knowledge_docs", []))
        seed_memory_store(memory_store, benchmark.get("memory_docs", []))

        retrieval_results = run_retrieval_benchmark(knowledge_store, memory_store, cases)
        tool_results = run_direct_tool_benchmark(cases)
        live_results, skip_reason = run_live_agent_benchmark(knowledge_store, memory_store, cases)

        summary = summarize_results(retrieval_results, tool_results, live_results)

        report = {
            "benchmark_file": str(benchmark_path),
            "summary": summary,
            "retrieval_results": retrieval_results,
            "tool_results": tool_results,
            "live_results": live_results,
            "skip_reason": skip_reason,
        }

        markdown_report = build_markdown_report(summary, retrieval_results, tool_results, live_results)

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        if args.markdown_output:
            markdown_path = Path(args.markdown_output)
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_path.write_text(markdown_report, encoding="utf-8")

        print(markdown_report)
        if skip_reason:
            print(skip_reason)
    finally:
        del knowledge_store
        del memory_store
        gc.collect()
        shutil.rmtree(temp_root, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())