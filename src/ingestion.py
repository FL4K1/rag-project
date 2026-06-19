"""ingestion.py — Document loading with OCR fallback and table extraction.

Phase 1: Load text from PDFs via PyPDFLoader.
Phase 2: 
  - Detect scanned pages and run OCR via pytesseract (if available).
  - Extract tables via pdfplumber and inject as separate Documents.
"""

import logging
import os
from glob import glob
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from config import RAGConfig

logger = logging.getLogger(__name__)


def load_documents(directory: str, cfg: RAGConfig = None) -> List[Document]:
    """Load all PDFs in *directory* with text extraction, OCR fallback, and table extraction.

    Args:
        directory: Path to the folder containing PDF files.
        cfg: RAGConfig instance. If None, uses defaults.

    Returns:
        List of Document objects with metadata: source_file, page, type.
    """
    if cfg is None:
        cfg = RAGConfig()

    all_docs: List[Document] = []
    pdf_paths = glob(os.path.join(directory, "*.pdf"))

    if not pdf_paths:
        logger.warning(f"No PDF files found in '{directory}'.")
        return []

    for pdf_path in pdf_paths:
        source_name = os.path.basename(pdf_path)
        logger.info(f"Loading '{source_name}' ...")

        # ── Step 1: Standard text extraction ─────────────────────────────
        try:
            loader = PyPDFLoader(pdf_path)
            loaded_pages: List[Document] = loader.load()
        except Exception as exc:
            logger.warning(f"PyPDFLoader failed for '{source_name}': {exc}")
            loaded_pages = []

        # Attach metadata
        for doc in loaded_pages:
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata["source_file"] = source_name
            doc.metadata["source"] = pdf_path
            doc.metadata.setdefault("type", "text")

        # ── Step 2: OCR fallback for scanned pages ────────────────────────
        if cfg.enable_ocr:
            _apply_ocr_fallback(pdf_path, source_name, loaded_pages, cfg)

        all_docs.extend(loaded_pages)

        # ── Step 3: Table extraction ──────────────────────────────────────
        if cfg.enable_table_extraction:
            from table_extractor import extract_tables, pdfplumber_available
            if pdfplumber_available():
                table_docs = extract_tables(pdf_path)
                all_docs.extend(table_docs)
                logger.info(f"  → {len(table_docs)} table document(s) extracted.")
            else:
                logger.warning("pdfplumber not available; skipping table extraction.")

    logger.info(f"Total documents loaded: {len(all_docs)}")
    return all_docs


def _apply_ocr_fallback(
    pdf_path: str,
    source_name: str,
    loaded_pages: List[Document],
    cfg: RAGConfig,
) -> None:
    """In-place: replace near-empty text pages with OCR results.

    Args:
        pdf_path: Path to the PDF.
        source_name: Filename (for metadata).
        loaded_pages: List of Documents already loaded by PyPDFLoader.
        cfg: RAGConfig instance (ocr_dpi used).
    """
    from ocr import is_scanned_page, ocr_pdf, ocr_available

    if not ocr_available():
        logger.warning(
            "OCR requested but dependencies missing (pdf2image / pytesseract). "
            "Skipping OCR fallback."
        )
        return

    # Determine which pages need OCR
    scanned_indices = [
        i for i, doc in enumerate(loaded_pages)
        if is_scanned_page(doc.page_content)
    ]

    if not scanned_indices:
        return

    logger.info(
        f"  → {len(scanned_indices)} scanned page(s) detected in '{source_name}'; running OCR ..."
    )

    ocr_texts = ocr_pdf(pdf_path, dpi=cfg.ocr_dpi)
    if not ocr_texts:
        return

    for idx in scanned_indices:
        if idx < len(ocr_texts) and ocr_texts[idx].strip():
            loaded_pages[idx].page_content = ocr_texts[idx]
            loaded_pages[idx].metadata["ocr"] = True
            logger.info(f"    OCR applied to page {idx}.")


def get_ingestion_stats(docs: List[Document]) -> dict:
    """Return a summary of ingested documents for display in the UI.

    Args:
        docs: List of loaded Documents.

    Returns:
        Dict with total_pages, table_count, ocr_page_count, source_files.
    """
    source_files = sorted({d.metadata.get("source_file", "unknown") for d in docs})
    table_count = sum(1 for d in docs if d.metadata.get("type") == "table")
    ocr_count = sum(1 for d in docs if d.metadata.get("ocr", False))
    text_pages = sum(1 for d in docs if d.metadata.get("type", "text") == "text")

    return {
        "source_files": source_files,
        "total_pages": text_pages,
        "table_count": table_count,
        "ocr_page_count": ocr_count,
    }
