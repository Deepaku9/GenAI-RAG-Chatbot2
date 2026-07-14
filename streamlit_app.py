import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

st.set_page_config(
    page_title="GenAI RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 GenAI RAG Chatbot")
st.caption("Powered by LangChain + FAISS + Ollama")

@st.cache_resource
def load_components():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = OllamaLLM(model="llama3.2")

    return retriever, llm

retriever, llm = load_components()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your documents...")

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the context below.

Context:

{context}

Question:

{question}

If the answer is not available in the context,
reply:

I don't know based on the provided documents.
"""

    response = llm.invoke(prompt)

    with st.chat_message("assistant"):

        st.markdown(response)

        with st.expander("📄 Source Documents"):

            for i, doc in enumerate(docs, start=1):

                st.markdown(f"### Source {i}")

                st.write(doc.metadata)

                st.write(doc.page_content)

                st.divider()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )