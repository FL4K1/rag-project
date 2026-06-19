"""table_extractor.py — Extract tables from PDFs using pdfplumber.

pdfplumber is pure Python (no Java/Ghostscript required).
Tables are converted to GitHub-flavored Markdown so the LLM can read them clearly.

Requires:
    pip install pdfplumber
"""

import logging
import os
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _table_to_markdown(table: List[List]) -> str:
    """Convert a pdfplumber table (list of rows, each row a list of cells) to Markdown."""
    if not table or not table[0]:
        return ""

    # Sanitize cells: replace None with empty string, strip whitespace
    clean = [[str(cell).strip() if cell is not None else "" for cell in row] for row in table]

    header = clean[0]
    separator = ["---"] * len(header)
    rows = clean[1:]

    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(separator) + " |")
    for row in rows:
        # Pad row to header width if needed
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(padded[:len(header)]) + " |")

    return "\n".join(lines)


def extract_tables(pdf_path: str) -> List[Document]:
    """Extract all tables from a PDF and return them as LangChain Documents.

    Each table becomes a separate Document with Markdown-formatted content and
    metadata: source_file, page, type='table', table_index.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of Document objects, one per table found.
    """
    try:
        import pdfplumber
    except ImportError:
        logger.warning(
            "pdfplumber not installed. Run 'pip install pdfplumber' to enable table extraction."
        )
        return []

    documents: List[Document] = []
    source_name = os.path.basename(pdf_path)
    table_count = 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    if not table:
                        continue
                    md = _table_to_markdown(table)
                    if not md.strip():
                        continue
                    table_count += 1
                    doc = Document(
                        page_content=f"[TABLE from {source_name}, Page {page_num + 1}]\n\n{md}",
                        metadata={
                            "source_file": source_name,
                            "source": pdf_path,
                            "page": page_num,
                            "type": "table",
                            "table_index": table_count,
                        },
                    )
                    documents.append(doc)
    except Exception as exc:
        logger.warning(f"Table extraction failed for '{pdf_path}': {exc}")

    logger.info(f"Extracted {table_count} table(s) from '{source_name}'.")
    return documents


def pdfplumber_available() -> bool:
    """Check whether pdfplumber is importable."""
    try:
        import pdfplumber  # noqa: F401
        return True
    except ImportError:
        return False
