"""logger.py — Append-only JSON-Lines query/answer log for the RAG pipeline.

Each query-answer cycle is recorded as a single JSON object per line
to `config.log_path` (default: logs/rag_log.json).

Log fields:
    timestamp       — ISO-8601 UTC timestamp
    query           — original user query
    expanded_query  — query after expansion (may equal query)
    num_docs        — number of docs fed to the LLM
    sources         — list of (source_file, page) pairs from retrieved docs
    answer_preview  — first 200 chars of the answer
    latency_ms      — total retrieval + generation time in milliseconds
    eval_scores     — dict with context_relevance, faithfulness, completeness
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def log_query(
    cfg,
    query: str,
    expanded_query: str,
    docs: List[Document],
    answer: str,
    latency_ms: float,
    eval_scores: Optional[Dict[str, float]] = None,
) -> None:
    """Append a single query-answer record to the JSON-Lines log file.

    Args:
        cfg: RAGConfig instance (provides log_path and enable_logging).
        query: Original user query string.
        expanded_query: Query after expansion (may equal original).
        docs: Retrieved + reranked documents passed to the LLM.
        answer: Final answer string produced by the LLM.
        latency_ms: End-to-end latency in milliseconds.
        eval_scores: Optional dict with evaluation metrics.
    """
    if not cfg.enable_logging:
        return

    log_path = cfg.log_path
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Collect unique sources
    sources = []
    seen = set()
    for doc in docs:
        key = (doc.metadata.get("source_file", "unknown"), doc.metadata.get("page", "?"))
        if key not in seen:
            seen.add(key)
            sources.append({"source_file": key[0], "page": key[1]})

    record: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "expanded_query": expanded_query,
        "num_docs": len(docs),
        "sources": sources,
        "answer_preview": answer[:200].replace("\n", " ") if answer else "",
        "latency_ms": round(latency_ms, 1),
        "eval_scores": eval_scores or {},
    }

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.warning(f"Failed to write to log '{log_path}': {exc}")


def read_logs(cfg) -> List[Dict[str, Any]]:
    """Read all log records from the JSON-Lines log file.

    Returns:
        List of dicts, newest first.
    """
    log_path = cfg.log_path
    if not os.path.exists(log_path):
        return []

    records = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception as exc:
        logger.warning(f"Failed to read log '{log_path}': {exc}")

    return list(reversed(records))  # newest first
