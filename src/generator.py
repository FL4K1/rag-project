from multiprocessing import context

from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3")

def generate_answer(query,docs):
    context = "" 

    for i, doc in enumerate(docs):
        context += f"[{i}] {doc.page_content}\n\n"

    prompt = f"""
You are a strict AI system.

Rules:
- Identify what the question is asking
- Only extract information related to that topic
- Ignore all other sections


If question is about datasets:
→ ONLY return dataset information
→ DO NOT return irrelevant or extra information

Context:
{context}

Question: {query}

Answer:
    """

    response = llm.invoke(prompt)
    return response
