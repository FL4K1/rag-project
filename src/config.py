from dataclasses import dataclass
from typing import List

@dataclass
class RAGConfig:
    # Chunking (Hierarchical)
    parent_chunk_size: int = 2000
    child_chunk_size: int = 400
    chunk_overlap: int = 100

    # Embedding model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Reranker model
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Retrieval parameters
    top_k: int = 10
    mmr_lambda: float = 0.7
    mmr_k: int = 10

    # Hybrid Score Fusion Weights
    vector_weight: float = 0.6
    bm25_weight: float = 0.4

    # Memory
    memory_window: int = 5

    # LLM Provider: "ollama" or "gemini"
    llm_provider: str = "ollama"

    # Ollama model for generation and query expansion
    ollama_model: str = "llama3.1:8b"

    # Gemini settings (used when llm_provider = "gemini")
    gemini_model: str = "gemini-2.0-flash"
    gemini_api_key: str = ""  # Set via UI or GEMINI_API_KEY environment variable

    # Logging
    log_path: str = "logs/rag_log.json"

    # Enable/disable features
    enable_query_expansion: bool = True
    enable_mmr: bool = True
    enable_streaming: bool = True
    enable_debug_mode: bool = False
    enable_logging: bool = True
    enable_eval_metrics: bool = True

    # Phase 2 — Document intelligence features
    enable_ocr: bool = True          # OCR for scanned/image-based PDF pages
    enable_table_extraction: bool = True  # Table extraction via pdfplumber

    # OCR settings
    ocr_engine: str = "pytesseract"
    pdf_to_image_backend: str = "pdf2image"
    ocr_dpi: int = 200

    # Table extraction
    table_extractor: str = "pdfplumber"
