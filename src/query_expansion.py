"""query_expansion.py — Expand short queries using the configured LLM provider."""

import logging
from config import RAGConfig

logger = logging.getLogger(__name__)


def expand_query(query: str, cfg: RAGConfig) -> str:
    """Expand short queries (<10 words) into a retrieval-oriented formulation.

    Uses the configured LLM provider (Ollama or Gemini) to add synonyms and
    related concepts. Only runs when:
      - cfg.enable_query_expansion is True
      - The query has fewer than 10 words

    Returns the original query on failure or if conditions are not met.
    """
    if not cfg.enable_query_expansion:
        return query
    if len(query.split()) >= 10:
        return query

    prompt = (
        "You are a retrieval query optimizer. Rewrite the following question to improve "
        "document retrieval by adding relevant synonyms, related technical terms, and "
        "domain-specific concepts. Output only the expanded question, nothing else.\n\n"
        f"Original question: {query}\n\nExpanded question:"
    )

    try:
        if cfg.llm_provider == "gemini":
            import os
            # pyrefly: ignore [missing-import]
            from langchain_google_genai import ChatGoogleGenerativeAI
            api_key = cfg.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                return query
            llm = ChatGoogleGenerativeAI(
                model=cfg.gemini_model,
                google_api_key=api_key,
                temperature=0.2,
                max_output_tokens=128,
            )
            response = llm.invoke(prompt)
            return response.content.strip() or query
        else:
            from langchain_ollama import OllamaLLM
            llm = OllamaLLM(model=cfg.ollama_model, temperature=0.2, num_predict=128)
            expanded = llm.invoke(prompt)
            return expanded.strip() or query

    except Exception as exc:
        logger.warning(f"Query expansion failed: {exc}")
        return query
