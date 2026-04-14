from ingestion import load_documents
from chunking import chunk_docs
from embedding import create_vector_store 
from retrieval import hybrid_retrieve, build_bm25, rerank
from generator import generate_answer
import os

# step 1: loading docs 
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample.pdf")
docs = load_documents(data_path)

# step 2: chunking docs
chunks = chunk_docs(docs)

# step 3: creating database 
db = create_vector_store(chunks) 

# step 4: building bm25 index
bm25 = build_bm25(chunks)

# step 5: query loop
while True: 
    query = input("Ask: " )

    docs = hybrid_retrieve(db, bm25, chunks, query)
    docs = rerank(query, docs)
    answer = generate_answer(query, docs)

    print("\nAnswer:\n", answer)

