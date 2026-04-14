import streamlit as st
import os
from ingestion import load_documents
from chunking import chunk_docs
from embedding import create_vector_store
from retrieval import hybrid_retrieve, build_bm25, rerank
from generator import generate_answer

# Page configuration
st.set_page_config(page_title="RAG Q&A System", layout="wide")

# Title
st.title("📚 RAG Q&A System")
st.markdown("Ask questions about your documents and get intelligent answers")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Load and process documents
    if st.button("Load Documents", use_container_width=True):
        with st.spinner("Loading documents..."):
            data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample.pdf")
            docs = load_documents(data_path)
            
            with st.spinner("Chunking documents..."):
                chunks = chunk_docs(docs)
            
            with st.spinner("Creating vector store..."):
                db = create_vector_store(chunks)
            
            with st.spinner("Building BM25 index..."):
                bm25 = build_bm25(chunks)
            
            # Store in session
            st.session_state.db = db
            st.session_state.bm25 = bm25
            st.session_state.chunks = chunks
            st.success("✅ Documents loaded successfully!")

# Initialize session state
if "db" not in st.session_state:
    st.info("👈 Click 'Load Documents' in the sidebar to get started")
else:
    # Main search interface
    st.subheader("Ask a Question")
    
    query = st.text_input("Enter your question:", placeholder="What is GPT-3?")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        search_button = st.button("Search", use_container_width=True)
    
    if search_button and query:
        with st.spinner("Searching and generating answer..."):
            # Retrieve documents
            retrieved_docs = hybrid_retrieve(
                st.session_state.db,
                st.session_state.bm25,
                st.session_state.chunks,
                query
            )
            
            # Rerank documents
            reranked_docs = rerank(query, retrieved_docs)
            
            # Generate answer
            answer = generate_answer(query, reranked_docs)
        
        # Display results
        st.subheader("Answer")
        st.write(answer)
        
        # Display retrieved documents
        with st.expander("📄 Retrieved Documents"):
            for i, doc in enumerate(reranked_docs, 1):
                st.markdown(f"**Document {i}:**")
                st.write(doc.page_content)
                st.divider()

# Footer
st.markdown("---")
st.caption("Powered by RAG (Retrieval-Augmented Generation)")
