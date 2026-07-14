from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_ollama import OllamaLLM

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k":3})

llm = OllamaLLM(model="llama3.2")

print("RAG Chatbot Ready")

print("------------------------")

while True:

    question = input("\nAsk: ")

    if question.lower() == "exit":
        break

    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are a helpful assistant.

Answer ONLY from the context below.

Context:

{context}

Question:

{question}

If the answer is not in the context,
say:

I don't know based on the provided documents.
"""

    response = llm.invoke(prompt)

    print("\nAnswer:\n")

    print(response)

    print("\nSources:")

    for doc in docs:

        print(doc.metadata)