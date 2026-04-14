from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder



# ============================| Building BM25 index once 
def build_bm25(chunks):
    corpus = [doc.page_content.split() for doc in chunks]
    return BM25Okapi(corpus)



# ============================| Mixed Hybid retrieval 
def hybrid_retrieve(db, bm25, chunks, query, k=6):
    # vector search
    vector_docs = db.similarity_search(query, k=k)

    # bm25 search 
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]



    bm25_docs = [chunks[i] for i in top_bm25_idx]

    # combining both results 
    combined = vector_docs + bm25_docs

    # removal of duplicates while preserving order
    unique_docs = list({doc.page_content: doc for doc in combined}.values())

    return unique_docs[:k]



# ============================| Reranking 
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
def rerank(query, docs, top_k=7):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)


    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, score in scored_docs[:top_k]]

