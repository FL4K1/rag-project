"""ocr.py — OCR-based text extraction for scanned/image-based PDF pages.

Requires:
    - pdf2image  (pip install pdf2image)
    - pytesseract (pip install pytesseract) + Tesseract binary
    - Pillow     (pip install Pillow)

If Tesseract is not installed, OCR silently returns empty string per page.
"""

import os
import logging
from typing import List

logger = logging.getLogger(__name__)

# Threshold: pages with fewer than this many characters are treated as scanned
SCANNED_PAGE_CHAR_THRESHOLD = 50


def is_scanned_page(page_text: str) -> bool:
    """Return True if the page appears to have no extractable text (likely scanned)."""
    return len(page_text.strip()) < SCANNED_PAGE_CHAR_THRESHOLD


def ocr_pdf(pdf_path: str, dpi: int = 200) -> List[str]:
    """Run OCR on every page of a PDF and return a list of text strings (one per page).

    Args:
        pdf_path: Absolute or relative path to the PDF file.
        dpi: Image resolution for pdf2image conversion (200 is fast & sufficient).

    Returns:
        List[str]: OCR'd text for each page. Empty string if OCR unavailable or page fails.
    """
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        logger.warning(
            "OCR libraries not found. Install pdf2image and pytesseract to enable OCR. "
            "Returning empty OCR results."
        )
        return []

    try:
        images = convert_from_path(pdf_path, dpi=dpi)
    except Exception as exc:
        logger.warning(f"pdf2image failed to convert '{pdf_path}': {exc}")
        return []

    results: List[str] = []
    for i, img in enumerate(images):
        try:
            text = pytesseract.image_to_string(img, lang="eng")
            results.append(text)
        except Exception as exc:
            logger.warning(f"pytesseract failed on page {i} of '{pdf_path}': {exc}")
            results.append("")

    return results


def ocr_available() -> bool:
    """Check whether OCR dependencies (pdf2image + pytesseract) are importable."""
    try:
        import pdf2image  # noqa: F401
        import pytesseract  # noqa: F401
        return True
    except ImportError:
        return False
