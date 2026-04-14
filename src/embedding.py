from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def create_vector_store(chunks):
    embedding_model = HuggingFaceEmbeddings(
        model_name = "ALL-MiniLM-L6-v2"
    
    )

    db = Chroma.from_documents(
        chunks, 
        embedding_model, 
        persist_directory = "embeddings/"

    )

    db.persist()
    return db 
