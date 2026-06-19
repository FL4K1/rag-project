"""chunking.py — Hierarchical chunking: parent → child with metadata enrichment."""

import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import RAGConfig


def chunk_docs(docs):
    """Perform hierarchical chunking into parent and child chunks.

    Each child chunk stores:
      - Its own text as page_content
      - parent_text: full text of the parent chunk (for parent-child retrieval)
      - parent_id: UUID linking child to parent
      - chunk_index: sequential index within the full document list
      - type: 'table' chunks bypass child splitting and are kept as-is

    Returns:
        List of child Document chunks (plus table docs unchanged).
    """
    cfg = RAGConfig()

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.parent_chunk_size,
        chunk_overlap=200,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.child_chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )

    # Separate text docs from table docs (tables skip hierarchical chunking)
    text_docs = [d for d in docs if d.metadata.get("type", "text") != "table"]
    table_docs = [d for d in docs if d.metadata.get("type") == "table"]

    parent_docs = parent_splitter.split_documents(text_docs)

    child_chunks = []
    chunk_index = 0

    for p_doc in parent_docs:
        p_id = str(uuid.uuid4())
        p_doc.metadata["doc_id"] = p_id

        c_docs = child_splitter.split_documents([p_doc])
        for c_doc in c_docs:
            c_doc.metadata["parent_id"] = p_id
            c_doc.metadata["parent_text"] = p_doc.page_content
            c_doc.metadata["chunk_index"] = chunk_index
            c_doc.metadata.setdefault("type", "text")
            chunk_index += 1
        child_chunks.extend(c_docs)

    # Table docs are kept whole — assign chunk_index for completeness
    for t_doc in table_docs:
        t_doc.metadata["chunk_index"] = chunk_index
        chunk_index += 1

    return child_chunks + table_docs
