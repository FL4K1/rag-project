from langchain_community.document_loaders import PyPDFLoader
def load_documents(path):
    loader = PyPDFLoader(path)
    docs = loader.load()
    return docs


