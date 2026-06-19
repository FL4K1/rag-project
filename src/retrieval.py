from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from copy import deepcopy
from config import RAGConfig


# ============================| Build BM25 index
def build_bm25(chunks):
    corpus = [doc.page_content.split() for doc in chunks]
    return BM25Okapi(corpus)


# ============================| Hybrid Score Fusion + Parent-Child Retrieval
def hybrid_retrieve(db, bm25, chunks, query, cfg: RAGConfig, source_filter=None):
    """Retrieve documents using vector similarity, BM25, optional MMR, and source filtering.

    Args:
        db: Chroma vector store.
        bm25: BM25 index built from ALL chunks (indices must match `chunks`).
        chunks: List[Document] — full corpus used for BM25 indexing.
        query: User query string.
        cfg: RAGConfig instance.
        source_filter: Optional list of source file paths to restrict results.
    Returns:
        List[Document]: Ranked documents after fusion, parent-child expansion, and MMR.
    """
    # 1. Vector search (retrieve from Chroma, then optionally filter by source)
    vector_results = db.similarity_search_with_relevance_scores(query, k=cfg.top_k)
    if source_filter:
        vector_results = [
            (doc, score) for doc, score in vector_results
            if doc.metadata.get("source") in source_filter
        ]

    # 2. BM25 search — always score over full corpus, then filter by source
    bm25_scores = bm25.get_scores(query.split())

    # Normalize BM25 scores to [0, 1]
    max_b = max(bm25_scores) if len(bm25_scores) > 0 else 1.0
    min_b = min(bm25_scores) if len(bm25_scores) > 0 else 0.0
    denom = max_b - min_b if max_b > min_b else 1.0

    top_bm25_idx = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:cfg.top_k]

    # 3. Fuse scores into candidate dict
    candidate_dict = {}

    for idx in top_bm25_idx:
        doc = chunks[idx]
        # Apply source filter before adding
        if source_filter and doc.metadata.get("source") not in source_filter:
            continue
        norm_b = (bm25_scores[idx] - min_b) / denom
        candidate_dict[doc.page_content] = {"doc": doc, "score": cfg.bm25_weight * norm_b}

    for doc, v_score in vector_results:
        weighted_v = cfg.vector_weight * v_score
        if doc.page_content in candidate_dict:
            candidate_dict[doc.page_content]["score"] += weighted_v
        else:
            candidate_dict[doc.page_content] = {"doc": doc, "score": weighted_v}

    # 4. Sort by fused score
    ranked_candidates = sorted(candidate_dict.values(), key=lambda x: x["score"], reverse=True)

    # 5. Parent-Child expansion: replace child text with parent context
    parent_docs = []
    seen_parents = set()
    for cand in ranked_candidates:
        doc = cand["doc"]
        parent_text = doc.metadata.get("parent_text", doc.page_content)
        if parent_text not in seen_parents:
            seen_parents.add(parent_text)
            parent_doc = deepcopy(doc)
            parent_doc.page_content = parent_text
            parent_doc.metadata.pop("parent_text", None)
            parent_docs.append(parent_doc)

    # 6. Apply MMR for diversity
    if cfg.enable_mmr:
        return apply_mmr(query, parent_docs, cfg.mmr_lambda, cfg.mmr_k)
    else:
        return parent_docs[:cfg.top_k]


def retrieve_with_expansion(db, bm25, chunks, query, cfg: RAGConfig, source_filter=None):
    """Retrieve using original query and, for short queries, also an expanded query.
    Results from both are fused and deduplicated.

    Returns:
        List[Document] and the expanded query string (or original if no expansion).
    """
    from query_expansion import expand_query

    docs_orig = hybrid_retrieve(db, bm25, chunks, query, cfg, source_filter)
    expanded_query = query

    if cfg.enable_query_expansion and len(query.split()) < 10:
        expanded_query = expand_query(query, cfg)
        if expanded_query and expanded_query != query:
            docs_exp = hybrid_retrieve(db, bm25, chunks, expanded_query, cfg, source_filter)
            # Fuse: deduplicate preserving original-query order first
            seen = {doc.page_content for doc in docs_orig}
            for doc in docs_exp:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    docs_orig.append(doc)

    return docs_orig, expanded_query


def apply_mmr(query, docs, lambda_param, k):
    """Select k diverse documents using Maximal Marginal Relevance.

    Args:
        query: query string (unused in word-overlap proxy, kept for future embedding MMR).
        docs: list of Document objects ranked by relevance.
        lambda_param: trade-off between relevance (1.0) and diversity (0.0).
        k: number of docs to return.
    Returns:
        List[Document] after MMR selection.
    """
    selected = []
    candidates = docs.copy()
    if not candidates:
        return []
    selected.append(candidates.pop(0))
    while len(selected) < k and candidates:
        scores = []
        for cand in candidates:
            relevance = 1.0  # proxy: original relevance order
            # Diversity: overlap with already-selected docs (higher overlap = less diverse)
            max_overlap = max(
                len(set(cand.page_content.split()) & set(sel.page_content.split()))
                for sel in selected
            )
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_overlap
            scores.append(mmr_score)
        best_idx = scores.index(max(scores))
        selected.append(candidates.pop(best_idx))
    return selected[:k]


# ============================| Cross-Encoder Reranking
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, docs, top_k=7):
    """Rerank documents using a cross-encoder model."""
    if not docs:
        return []
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:top_k]]
