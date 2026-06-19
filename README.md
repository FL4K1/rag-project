# 📚 RAG Q&A System

A production-ready **Retrieval-Augmented Generation (RAG)** system that lets you query PDF documents and receive intelligent, cited, structured answers. Built with Python, LangChain, ChromaDB, and Streamlit.

---

## ✨ Features

### Document Intelligence (Phase 2)
- **Multi-PDF ingestion** — upload and index multiple PDFs simultaneously
- **OCR fallback** — detects scanned/image-based pages and runs Tesseract OCR automatically
- **Table extraction** — extracts PDF tables as Markdown via `pdfplumber` (no Java required)
- **Hierarchical chunking** — parent→child chunking strategy preserves full context during retrieval

### Retrieval Pipeline
- **Hybrid search** — combines ChromaDB vector search with BM25 keyword search
- **Query expansion** — short queries are automatically expanded using the LLM
- **Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2` reranks results for precision
- **MMR diversity** — Maximal Marginal Relevance ensures non-redundant retrieved chunks
- **Source filtering** — filter answers to specific uploaded documents

### LLM Generation
- **Dual provider support** — switch between local Ollama and Google Gemini from the UI
  - 🖥️ **Ollama** (local, private): `llama3.1:8b`, `llama3:8b`, `mistral:7b`
  - ✨ **Gemini** (cloud, free tier): `gemini-2.0-flash`, `gemini-1.5-flash-latest`, `gemini-1.5-pro-latest`
- **Streaming output** — token-by-token streaming for both providers
- **Conversation memory** — retains last 5 exchanges for contextual follow-up questions

### Observability (Phase 3)
- **RAG Triad evaluation** — per-answer scores for Context Relevance, Faithfulness, and Completeness
- **Query logging** — every query/answer logged to `logs/rag_log.json` with latency and eval scores
- **Answer latency** — end-to-end timing displayed per response

### UI Features
- **📋 Copy response** — one-click clipboard copy for every answer
- **💾 Export chat** — download full conversation as `.txt`
- **🗑️ Clear chat** — reset conversation history
- **📎 Source citations** — file name + page number shown below every answer
- **📁 Document stats** — shows page count, table count, OCR'd pages per upload

---

## 🗂️ Project Structure

```
RAG_Project/
├── src/
│   ├── app.py               # Streamlit web UI
│   ├── main.py              # CLI interface
│   ├── config.py            # All configuration & feature flags
│   ├── ingestion.py         # PDF loading + OCR fallback + table extraction
│   ├── chunking.py          # Hierarchical parent→child chunking
│   ├── embedding.py         # ChromaDB vector store creation
│   ├── retrieval.py         # Hybrid search, MMR, reranking
│   ├── query_expansion.py   # LLM-based query expansion
│   ├── generator.py         # Ollama + Gemini answer generation
│   ├── citations.py         # Source citation formatting
│   ├── evaluator.py         # RAG Triad evaluation metrics
│   ├── logger.py            # JSON-Lines query/answer logging
│   ├── ocr.py               # Tesseract OCR for scanned PDFs
│   └── table_extractor.py   # pdfplumber table → Markdown extraction
├── data/                    # Place your PDF files here (gitignored)
├── embeddings/              # ChromaDB vector store (auto-generated, gitignored)
├── logs/                    # Query logs (auto-generated, gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) (for local LLM) — or a free [Gemini API key](https://aistudio.google.com/app/apikey)

### 2. Install dependencies
```bash
git clone https://github.com/FL4K1/rag-project.git
cd rag-project
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 3. Pull an Ollama model (if using local LLM)
```bash
ollama pull llama3.1:8b
```

### 4. Run the app
```bash
# Terminal 1 — start Ollama server (skip if using Gemini)
ollama serve

# Terminal 2 — launch the web app
streamlit run src/app.py
```

Open **http://localhost:8501** in your browser.

---

## ⚙️ Configuration

All settings live in [`src/config.py`](src/config.py):

| Flag | Default | Description |
|------|---------|-------------|
| `llm_provider` | `"ollama"` | `"ollama"` or `"gemini"` |
| `ollama_model` | `"llama3.1:8b"` | Ollama model name |
| `gemini_model` | `"gemini-2.0-flash"` | Gemini model name |
| `gemini_api_key` | `""` | Set here or enter in the sidebar UI |
| `enable_ocr` | `True` | OCR for scanned PDF pages |
| `enable_table_extraction` | `True` | Extract tables as Markdown |
| `enable_logging` | `True` | Log queries to `logs/rag_log.json` |
| `enable_eval_metrics` | `True` | Show RAG Triad scores per answer |
| `enable_query_expansion` | `True` | Expand short queries using LLM |
| `enable_streaming` | `True` | Stream tokens in real time |

### Gemini API Key
Get a **free** key at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).  
Enter it in the sidebar at runtime, or set it in `config.py`:
```python
gemini_api_key: str = "YOUR_KEY_HERE"
```

---

## 📦 Key Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` + ecosystem | Core RAG orchestration |
| `chromadb` | Vector store |
| `sentence-transformers` | Embeddings (`all-MiniLM-L6-v2`) |
| `rank_bm25` | BM25 keyword search |
| `langchain-ollama` | Local LLM integration |
| `langchain-google-genai` | Gemini API integration |
| `pdfplumber` | Table extraction |
| `pdf2image` + `pytesseract` | OCR for scanned PDFs |
| `streamlit` | Web UI |

> **Note:** OCR requires the [Tesseract binary](https://github.com/UB-Mannheim/tesseract/wiki) installed separately on Windows. Without it, OCR is gracefully skipped.
