# RAG Q&A System

This project is a complete end-to-end Retrieval-Augmented Generation (RAG) system built with Python. It allows users to query PDF documents and receive intelligent, context-aware answers. The system leverages hybrid search techniques, document reranking, and local Large Language Models (LLMs) to ensure high-quality responses.

## Features

- **Document Ingestion**: Seamlessly load and process PDF documents.
- **Advanced Chunking**: Uses recursive character splitting to divide documents into manageable overlapping chunks.
- **Hybrid Retrieval**: Combines semantic vector search (ChromaDB + HuggingFace Embeddings) with keyword-based search (BM25) for robust document retrieval.
- **Cross-Encoder Reranking**: Refines and reranks retrieved documents using `ms-marco-MiniLM-L-6-v2` to ensure the most relevant context is fed to the LLM.
- **Local LLM Integration**: Generates answers using Ollama with the `llama3` model, ensuring privacy and local execution.
- **Interactive UI**: A user-friendly web interface built with Streamlit for uploading documents and asking questions.
- **CLI Support**: A command-line interface is also available for quick terminal-based interactions.

## Architecture & Modules

The source code (`src/`) is organized into modular components:

- `app.py`: The Streamlit web application providing the user interface.
- `main.py`: The Command Line Interface (CLI) application for terminal-based interactions.
- `ingestion.py`: Handles loading documents (e.g., from PDF) using LangChain's `PyPDFLoader`.
- `chunking.py`: Splits loaded documents into optimal chunks using `RecursiveCharacterTextSplitter`.
- `embedding.py`: Generates vector embeddings for chunks using `ALL-MiniLM-L6-v2` and stores them in a local `Chroma` database (`embeddings/`).
- `retrieval.py`: Implements the hybrid retrieval logic (Vector + BM25) and cross-encoder reranking.
- `generator.py`: Constructs the prompt context and queries the local Ollama LLM (`llama3`) for the final answer.

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/) installed and running locally with the `llama3` model pulled (`ollama run llama3`).
- Required Python packages (see `requirement.txt`).

## Installation

1. Clone or navigate to the project directory:
   ```bash
   cd RAG_Project
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirement.txt
   ```

   *Dependencies include:*
   - `langchain`, `langchain-community`, `langchain-openai`, `langchain-text-splitters`
   - `chromadb`
   - `sentence-transformers`
   - `rank_bm25` (Required for BM25 search, install if missing: `pip install rank_bm25`)
   - `langchain-ollama` (Required for Ollama integration, install if missing: `pip install langchain-ollama`)
   - `streamlit` (Required for the web app, install if missing: `pip install streamlit`)

## Usage

Ensure you have a sample PDF document located at `data/sample.pdf` before running the application.

### Streamlit Web Interface

To launch the interactive web application:

```bash
streamlit run src/app.py
```
1. Open the provided local URL in your browser.
2. Click "Load Documents" in the sidebar to process the PDF.
3. Enter your questions in the main interface and receive answers along with the retrieved source documents.

### Command Line Interface

To run the application in the terminal:

```bash
python src/main.py
```
The script will automatically load the documents, build the indexes, and start an interactive query loop where you can ask questions.

## Directory Structure

```text
RAG_Project/
├── data/               # Directory for source documents (e.g., sample.pdf)
├── embeddings/         # Persisted ChromaDB vector store
├── src/                # Python source code
│   ├── app.py          # Streamlit UI
│   ├── chunking.py     # Document splitting
│   ├── embedding.py    # Vector store creation
│   ├── generator.py    # LLM answer generation
│   ├── ingestion.py    # Document loading
│   ├── main.py         # CLI application
│   └── retrieval.py    # Hybrid search & reranking
├── requirement.txt     # Python dependencies
└── README.md           # Project documentation
```
