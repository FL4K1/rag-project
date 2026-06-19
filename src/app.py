"""app.py — Streamlit web interface for the RAG Q&A System.

Phase 1: Core chat UI with document upload, hybrid retrieval, streaming answers.
Phase 3: Eval metrics display, document ingestion stats, chat export, clear chat,
         query history panel, per-answer latency display.
"""

import time
import streamlit as st
from pathlib import Path

from ingestion import load_documents, get_ingestion_stats
from chunking import chunk_docs
from embedding import create_vector_store
from retrieval import retrieve_with_expansion, build_bm25, rerank
from generator import generate_answer
from citations import format_citations
from evaluator import evaluate
from logger import log_query, read_logs
from config import RAGConfig

# ── Configuration ─────────────────────────────────────────────────────────────
cfg = RAGConfig()

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Metric card styling */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 8px 12px;
    }
    /* Score colour coding via data attributes */
    .score-high { color: #22c55e; font-weight: bold; }
    .score-mid  { color: #f59e0b; font-weight: bold; }
    .score-low  { color: #ef4444; font-weight: bold; }
    /* Source badge */
    .source-badge {
        display: inline-block;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.4);
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_all_sources(chunks):
    """Return sorted list of unique source file paths from chunk metadata."""
    sources = set()
    for doc in chunks:
        src = doc.metadata.get("source")
        if src:
            sources.add(src)
    return sorted(sources)


def score_color(score: float) -> str:
    """Return an HTML-coloured score string."""
    cls = "score-high" if score >= 0.65 else ("score-mid" if score >= 0.35 else "score-low")
    return f'<span class="{cls}">{score:.2f}</span>'


def export_chat_txt(chat_history) -> str:
    """Convert chat history to plain text for download."""
    lines = ["RAG Q&A — Conversation Export\n" + "=" * 40]
    for role, msg in chat_history:
        prefix = "You" if role == "user" else "Assistant"
        lines.append(f"\n[{prefix}]\n{msg}")
    return "\n".join(lines)


# ── Header ────────────────────────────────────────────────────────────────────
st.title("📚 RAG Q&A System")
st.markdown(
    "Ask questions about your documents and get **intelligent, cited answers** "
    "powered by hybrid retrieval and local LLMs."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    # ── LLM Provider ─────────────────────────────────────────────────────
    st.subheader("🤖 LLM Provider")
    provider = st.radio(
        "Select model backend",
        options=["ollama", "gemini"],
        format_func=lambda x: "🖥️ Ollama (Local)" if x == "ollama" else "✨ Gemini (Free API)",
        index=0,
        horizontal=True,
        key="llm_provider_radio",
    )
    cfg.llm_provider = provider

    if provider == "ollama":
        ollama_model_choice = st.selectbox(
            "Ollama model",
            options=["llama3.1:8b", "llama3:8b", "mistral:7b", "llama3:70b"],
            index=0,
            help="Must be pulled via: ollama pull <model>",
        )
        cfg.ollama_model = ollama_model_choice
        st.caption(f"Using local Ollama · `{cfg.ollama_model}`")
    else:
        gemini_model_choice = st.selectbox(
            "Gemini model",
            options=["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash-latest", "gemini-1.5-pro-latest"],
            index=0,
            help="gemini-2.0-flash is the recommended free-tier model.",
        )
        cfg.gemini_model = gemini_model_choice
        api_key_input = st.text_input(
            "🔑 Gemini API Key",
            type="password",
            placeholder="Paste your API key here…",
            help="Get a free key at https://aistudio.google.com/app/apikey",
        )
        if api_key_input:
            cfg.gemini_api_key = api_key_input
            st.success("API key set ✓")
        elif not cfg.gemini_api_key:
            st.warning("⚠️ No API key — answers will fail until one is entered.")
        st.caption(f"Using Google Gemini · `{cfg.gemini_model}`")

    st.divider()

    # PDF upload
    uploaded_files = st.file_uploader(
        "Upload PDF documents", type=["pdf"], accept_multiple_files=True,
        key="pdf_uploader",
    )
    rebuild = st.button("🔄 Rebuild Index", use_container_width=True)

    # Source filter
    source_filter = []
    if "chunks" in st.session_state and st.session_state.chunks:
        source_options = get_all_sources(st.session_state.chunks)
        if source_options:
            source_filter = st.multiselect(
                "🔍 Filter sources", options=source_options,
                help="Leave empty to search all documents.",
            )

    # Feature toggles
    st.divider()
    st.subheader("Display Options")
    show_sources   = st.checkbox("📎 Show source citations", value=True)
    show_metrics   = st.checkbox("📊 Show eval metrics per answer", value=True)
    show_latency   = st.checkbox("⏱️ Show answer latency", value=True)
    debug_mode     = st.checkbox("🐛 Debug mode (expanded query)", value=False)

    st.divider()

    # Chat controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.eval_history = []
            st.rerun()
    with col2:
        if "chat_history" in st.session_state and st.session_state.chat_history:
            chat_text = export_chat_txt(st.session_state.chat_history)
            st.download_button(
                "💾 Export Chat",
                data=chat_text,
                file_name="rag_conversation.txt",
                mime="text/plain",
                use_container_width=True,
            )

    st.divider()

    # ── Index build ───────────────────────────────────────────────────────
    if uploaded_files or rebuild:
        if not uploaded_files and not st.session_state.get("db"):
            st.warning("No files uploaded.")
        else:
            data_dir = Path(__file__).parent.parent / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            if uploaded_files:
                with st.spinner("💾 Saving uploaded files..."):
                    for uf in uploaded_files:
                        (data_dir / uf.name).write_bytes(uf.getvalue())

            with st.spinner("📄 Loading documents (OCR + tables)..."):
                docs = load_documents(str(data_dir), cfg)

            if not docs:
                st.error("No PDF documents found. Please upload at least one PDF.")
            else:
                stats = get_ingestion_stats(docs)

                with st.spinner("✂️ Chunking documents..."):
                    chunks = chunk_docs(docs)
                with st.spinner("🧠 Building vector store..."):
                    db = create_vector_store(chunks)
                with st.spinner("📊 Building BM25 index..."):
                    bm25 = build_bm25(chunks)

                st.session_state.db = db
                st.session_state.bm25 = bm25
                st.session_state.chunks = chunks
                st.session_state.ingestion_stats = stats
                st.session_state.chat_history = []
                st.session_state.eval_history = []

                st.success(
                    f"✅ Indexed {len(stats['source_files'])} file(s) · "
                    f"{stats['total_pages']} pages · "
                    f"{stats['table_count']} tables · "
                    f"{len(chunks)} chunks"
                )
                if stats["ocr_page_count"] > 0:
                    st.info(f"🔍 OCR applied to {stats['ocr_page_count']} scanned page(s).")

    # ── Ingestion stats panel ─────────────────────────────────────────────
    if "ingestion_stats" in st.session_state:
        st.divider()
        st.subheader("📁 Loaded Documents")
        s = st.session_state.ingestion_stats
        for fname in s["source_files"]:
            st.markdown(f"• `{fname}`")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Pages", s["total_pages"])
        col_b.metric("Tables", s["table_count"])
        col_c.metric("OCR'd", s["ocr_page_count"])


# ── Main chat area ─────────────────────────────────────────────────────────────
if "db" not in st.session_state:
    st.info("👈 Upload PDFs and click **Rebuild Index** in the sidebar to get started.")
    st.stop()

# Ensure session state initialised
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "eval_history" not in st.session_state:
    st.session_state.eval_history = []

# Render existing conversation with stored eval data
for i, (role, msg) in enumerate(st.session_state.chat_history):
    st.chat_message(role).write(msg)
    # Show stored metrics below assistant turns
    if role == "assistant" and show_metrics and i < len(st.session_state.eval_history):
        ev = st.session_state.eval_history[i // 2]  # one eval per Q-A pair
        if ev:
            with st.expander("📊 Answer Quality Metrics", expanded=False):
                c1, c2, c3 = st.columns(3)
                c1.metric("Context Relevance", f"{ev['context_relevance']:.0%}")
                c2.metric("Faithfulness",       f"{ev['faithfulness']:.0%}")
                c3.metric("Completeness",        f"{ev['completeness']:.0%}")
                if show_latency and "latency_ms" in ev:
                    st.caption(f"⏱️ Latency: {ev['latency_ms']:.0f} ms")

# ── Chat input ─────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.chat_history.append(("user", user_input))

    t_start = time.monotonic()

    with st.spinner("🔍 Retrieving relevant context..."):
        retrieved_docs, expanded_query = retrieve_with_expansion(
            st.session_state.db,
            st.session_state.bm25,
            st.session_state.chunks,
            user_input,
            cfg,
            source_filter if source_filter else None,
        )
        reranked_docs = rerank(user_input, retrieved_docs)

    if debug_mode and expanded_query != user_input:
        st.sidebar.info(f"**Expanded query:**\n{expanded_query}")

    # Generate and stream answer
    answer_gen = generate_answer(user_input, reranked_docs, st.session_state.chat_history, cfg=cfg)

    with st.chat_message("assistant"):
        if cfg.enable_streaming and hasattr(answer_gen, "__iter__") and not isinstance(answer_gen, str):
            final_answer = st.write_stream(answer_gen)
        else:
            final_answer = answer_gen
            st.write(final_answer)

    latency_ms = (time.monotonic() - t_start) * 1000

    # Citations
    _, citations = format_citations(final_answer, reranked_docs)
    if show_sources and citations:
        st.caption(citations)

    # Copy-to-clipboard button
    if final_answer:
        escaped = (final_answer or "").replace("`", "\\`").replace("\\", "\\\\").replace("\n", "\\n")
        st.components.v1.html(f"""
        <button id="copy-btn" onclick="
            navigator.clipboard.writeText(`{escaped}`).then(() => {{
                var btn = document.getElementById('copy-btn');
                btn.innerText = '✅ Copied!';
                btn.style.background = '#22c55e22';
                btn.style.borderColor = '#22c55e';
                btn.style.color = '#22c55e';
                setTimeout(() => {{
                    btn.innerText = '📋 Copy Response';
                    btn.style.background = '';
                    btn.style.borderColor = '';
                    btn.style.color = '';
                }}, 2000);
            }});
        " style="
            cursor:pointer;
            padding:5px 14px;
            border:1px solid #555;
            border-radius:6px;
            background:transparent;
            color:#ccc;
            font-size:0.8rem;
            transition: all 0.2s ease;
        ">📋 Copy Response</button>
        """, height=40)

    # Eval metrics
    ev_scores = {}
    if cfg.enable_eval_metrics:
        ev_scores = evaluate(user_input, final_answer or "", reranked_docs)
        ev_scores["latency_ms"] = latency_ms

        if show_metrics:
            with st.expander("📊 Answer Quality Metrics", expanded=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("Context Relevance", f"{ev_scores['context_relevance']:.0%}",
                          help="How on-topic the retrieved chunks are for your query.")
                c2.metric("Faithfulness",       f"{ev_scores['faithfulness']:.0%}",
                          help="How much of the answer is grounded in the retrieved sources.")
                c3.metric("Completeness",        f"{ev_scores['completeness']:.0%}",
                          help="Heuristic answer length/detail score.")
                if show_latency:
                    st.caption(f"⏱️ Total latency: **{latency_ms:.0f} ms**")

    # Logging
    if cfg.enable_logging:
        log_query(
            cfg=cfg,
            query=user_input,
            expanded_query=expanded_query,
            docs=reranked_docs,
            answer=final_answer or "",
            latency_ms=latency_ms,
            eval_scores={k: v for k, v in ev_scores.items() if k != "latency_ms"},
        )

    # Store reply and eval scores
    st.session_state.chat_history.append(("assistant", final_answer))
    st.session_state.eval_history.append(ev_scores)

    # Enforce memory window
    max_len = cfg.memory_window * 2
    if len(st.session_state.chat_history) > max_len:
        st.session_state.chat_history = st.session_state.chat_history[-max_len:]
        st.session_state.eval_history = st.session_state.eval_history[-(max_len // 2):]

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Powered by **RAG** · LangChain · ChromaDB · Ollama (`llama3.1:8b`) · "
    "pdfplumber · sentence-transformers"
)
