from typing import List
from langchain_core.documents import Document


def format_citations(answer: str, docs: List[Document]):
    """Create a citation string and return answer with citations.

    Args:
        answer: Generated answer text.
        docs: List of retrieved Document objects.
    Returns:
        Tuple[str, str]: (answer_text, citation_text) where citation_text contains a markdown list of sources.
    """
    # Build a set of unique (source_file, page) pairs
    citations = []
    seen = set()
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "?")
        key = (source, page)
        if key not in seen:
            seen.add(key)
            citations.append(f"* {source} (Page {page})")
    citation_section = "\n**Sources:**\n" + "\n".join(citations) if citations else ""
    return answer, citation_section
