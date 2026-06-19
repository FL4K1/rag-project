"""evaluator.py — Lightweight RAG Triad evaluation metrics.

Three metrics are computed locally (no external API calls):

1. Context Relevance  — How relevant are retrieved chunks to the query?
   Method: Token overlap (Jaccard similarity) between query tokens and
           the combined context tokens. Score ∈ [0, 1].

2. Answer Faithfulness — Is the answer grounded in the retrieved context?
   Method: Fraction of answer bigrams present in the context.
   Score ∈ [0, 1]. High score = answer comes from context.

3. Answer Completeness — Does the answer seem thorough?
   Method: Normalized length score. Answers < 50 words score low;
           answers ≥ 150 words score near 1.0.
   Score ∈ [0, 1].

These are heuristic proxies — not ground-truth evaluations — but they are
cheap, fast, and useful for spotting regressions or poor retrievals.
"""

from typing import Dict, List
from langchain_core.documents import Document


def _tokenize(text: str) -> set:
    """Lowercase word-token set, stripping punctuation."""
    import re
    return set(re.findall(r"\b[a-z]{2,}\b", text.lower()))


def _bigrams(text: str) -> set:
    """Return set of adjacent word pairs from text."""
    import re
    words = re.findall(r"\b[a-z]{2,}\b", text.lower())
    return set(zip(words, words[1:]))


def context_relevance(query: str, docs: List[Document]) -> float:
    """Jaccard similarity between query tokens and all context tokens.

    Args:
        query: User query string.
        docs: Retrieved documents.

    Returns:
        float in [0, 1]. Higher = context is more on-topic for the query.
    """
    if not docs:
        return 0.0
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0

    context_text = " ".join(doc.page_content for doc in docs)
    context_tokens = _tokenize(context_text)

    intersection = query_tokens & context_tokens
    union = query_tokens | context_tokens
    return round(len(intersection) / len(union), 3) if union else 0.0


def answer_faithfulness(answer: str, docs: List[Document]) -> float:
    """Fraction of answer bigrams that appear in the retrieved context.

    Args:
        answer: LLM-generated answer string.
        docs: Retrieved documents used as context.

    Returns:
        float in [0, 1]. Higher = answer is more grounded in the sources.
    """
    if not answer or not docs:
        return 0.0

    answer_bigrams = _bigrams(answer)
    if not answer_bigrams:
        return 0.0

    context_text = " ".join(doc.page_content for doc in docs)
    context_bigrams = _bigrams(context_text)

    overlap = answer_bigrams & context_bigrams
    return round(len(overlap) / len(answer_bigrams), 3)


def answer_completeness(answer: str) -> float:
    """Heuristic completeness score based on answer word count.

    < 30 words  → low (0.0–0.4)
    30–150 words → medium (0.4–0.8)
    > 150 words → high (0.8–1.0)

    Args:
        answer: LLM-generated answer string.

    Returns:
        float in [0, 1].
    """
    if not answer:
        return 0.0
    word_count = len(answer.split())
    # Sigmoid-like scaling capped at 1.0
    score = min(1.0, word_count / 150)
    return round(score, 3)


def evaluate(query: str, answer: str, docs: List[Document]) -> Dict[str, float]:
    """Run all three RAG Triad metrics and return a dict of scores.

    Args:
        query: Original user query.
        answer: LLM-generated answer.
        docs: Retrieved + reranked documents.

    Returns:
        Dict with keys: context_relevance, faithfulness, completeness.
    """
    return {
        "context_relevance": context_relevance(query, docs),
        "faithfulness": answer_faithfulness(answer, docs),
        "completeness": answer_completeness(answer),
    }
