import os
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import DATA_PATH, EMBEDDING_MODEL, HUGGINGFACE_TOKEN

if HUGGINGFACE_TOKEN:
    login(token=HUGGINGFACE_TOKEN)

# SETUP MODEL EMBEDDING
def get_embedding_function():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return embeddings


# LOAD DOCUMENT
def load_documents():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        return []
    loaders = {
        ".pdf": PyMuPDFLoader,
        ".txt": TextLoader,
        ".docx": Docx2txtLoader,
    }
    documents = []
    for filename in os.listdir(DATA_PATH):
        ext = os.path.splitext(filename)[1]
        if ext in loaders:
            loader_class = loaders[ext]
            file_path = os.path.join(DATA_PATH, filename)
            loader = loader_class(file_path)
            documents.extend(loader.load())
    return documents


# CHUNKING DOCUMENT
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


# SAVE TO VECTOR DATABASE
def create_vectorstore(chunks):
    """Create in-memory vectorstore and return it"""
    if not chunks:
        return None

    vectorstore = Chroma.from_documents(
        chunks,
        get_embedding_function(),
    )
    return vectorstore