from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import RAGConfig

cfg = RAGConfig()

def create_vector_store(chunks):
    """Create (or overwrite) a Chroma vector store from document chunks."""
    embedding_model = HuggingFaceEmbeddings(model_name=cfg.embedding_model)

    db = Chroma.from_documents(
        chunks,
        embedding_model,
        persist_directory="embeddings/",
    )
    return db
