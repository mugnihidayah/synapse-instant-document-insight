import os
import tempfile
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import EMBEDDING_MODEL, HUGGINGFACE_TOKEN

if HUGGINGFACE_TOKEN:
    os.environ["HF_TOKEN"] = HUGGINGFACE_TOKEN

# SETUP MODEL EMBEDDING
def get_embedding_function():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return embeddings


# LOAD DOCUMENT
import tempfile

def load_documents_from_files(uploaded_files):
    """Load documents directly from uploaded file objects (in-memory)"""
    loaders = {
        ".pdf": PyMuPDFLoader,
        ".txt": TextLoader,
        ".docx": Docx2txtLoader,
    }
    
    documents = []
    for uploaded_file in uploaded_files:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext in loaders:
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            
            # Load from temp file
            loader_class = loaders[ext]
            loader = loader_class(tmp_path)
            docs = loader.load()
            
            # Update metadata with original file name and page +1
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
                if "page" in doc.metadata:
                    doc.metadata["page"] = doc.metadata["page"] + 1
            
            documents.extend(docs)
            
            # Delete temp file
            os.unlink(tmp_path)
    
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