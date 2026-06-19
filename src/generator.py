"""generator.py — LLM answer generation supporting Ollama (local) and Gemini (cloud).

Provider is selected via cfg.llm_provider:
  - "ollama"  → uses langchain_ollama.OllamaLLM  (local, private, free)
  - "gemini"  → uses langchain_google_genai.ChatGoogleGenerativeAI (cloud, free-tier API key)
"""

import os
import logging
from config import RAGConfig

logger = logging.getLogger(__name__)


def _build_prompt(query: str, docs, chat_history, cfg: RAGConfig) -> str:
    """Construct the full prompt string from query, docs, and chat history."""
    history_text = ""
    if chat_history:
        window = chat_history[-(cfg.memory_window * 2):]
        for role, msg in window:
            prefix = "User:" if role == "user" else "Assistant:"
            history_text += f"{prefix} {msg}\n"

    context = ""
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "?")
        doc_type = doc.metadata.get("type", "text")
        type_label = " [TABLE]" if doc_type == "table" else ""
        context += f"[Source {i}{type_label} | {source}, p.{page}]:\n{doc.page_content}\n\n"

    history_section = f"\n### Conversation History:\n{history_text}\n" if history_text else ""

    return f"""You are an expert AI research assistant with deep analytical capabilities. \
Your role is to provide comprehensive, insightful, and well-structured answers based on the provided context.

### Instructions:
- Provide a **thorough, detailed answer** — do not give one-liners. Elaborate on key concepts.
- **Structure your response clearly** using sections, bullet points, or numbered lists where appropriate.
- **Cite your sources** inline using [Source N] notation whenever you reference specific information.
- **Explain the "why" and "how"**, not just the "what" — give reasoning, implications, and context.
- If multiple perspectives or aspects exist, **cover each one** systematically.
- If the context is insufficient to fully answer, clearly state what is known and what is missing.
- Use **plain, professional language** that is easy to understand but intellectually rich.
- End with a **brief summary or key takeaway** if the answer is long.
{history_section}
### Context:
{context}

### Question:
{query}

### Answer:
"""


def _get_ollama_llm(cfg: RAGConfig):
    """Return a configured OllamaLLM instance."""
    from langchain_ollama import OllamaLLM
    return OllamaLLM(model=cfg.ollama_model, temperature=0.3, num_predict=1024)


def _get_gemini_llm(cfg: RAGConfig):
    """Return a configured Google Gemini chat model instance."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = cfg.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "Gemini API key not set. Enter it in the sidebar or set the "
            "GEMINI_API_KEY environment variable."
        )
    return ChatGoogleGenerativeAI(
        model=cfg.gemini_model,
        google_api_key=api_key,
        temperature=0.3,
        max_output_tokens=1024,
        streaming=cfg.enable_streaming,
    )


def generate_answer(query: str, docs, chat_history=None, cfg: RAGConfig = None):
    """Generate an answer using the configured LLM provider.

    Args:
        query: User question string.
        docs: List of retrieved Document objects.
        chat_history: List of (role, message) tuples for conversation memory.
        cfg: RAGConfig instance. If None, uses defaults.

    Returns:
        str or generator: Generated answer text, or a streaming generator.
    """
    if cfg is None:
        cfg = RAGConfig()

    prompt = _build_prompt(query, docs, chat_history, cfg)

    if cfg.llm_provider == "gemini":
        llm = _get_gemini_llm(cfg)
        if cfg.enable_streaming:
            # Gemini streaming returns AIMessageChunk objects; yield .content strings
            def _gemini_stream():
                for chunk in llm.stream(prompt):
                    yield chunk.content
            return _gemini_stream()
        else:
            response = llm.invoke(prompt)
            return response.content

    else:  # default: ollama
        llm = _get_ollama_llm(cfg)
        if cfg.enable_streaming and hasattr(llm, "stream"):
            return llm.stream(prompt)
        else:
            return llm.invoke(prompt)
