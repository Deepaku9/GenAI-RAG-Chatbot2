import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_FOLDER = "data"

documents = []

for file in os.listdir(DATA_FOLDER):

    if file.endswith(".pdf"):

        loader = PyPDFLoader(os.path.join(DATA_FOLDER, file))

        documents.extend(loader.load())

print(f"Loaded {len(documents)} pages")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(chunks, embeddings)

db.save_local("vectorstore")

print("Vector DB created successfully")