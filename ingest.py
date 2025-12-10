import os
import shutil
import chromadb
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import (
  DATA_PATH,
  DB_PATH,
  EMBEDDING_MODEL,
)

# SETUP MODEL EMBEDDING
def get_embedding_function():
  print("Initializing the Multilingual Embedding Model...")
  embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
  return embeddings

# LOAD DOCUMENT
def load_documents():
  print("Loading documents from the data folder...")
  if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
    print(f"The {DATA_PATH} folder was not found. Creating a new folder. Please fill in the document.")
    return []

  loaders = {
    '.pdf': PyMuPDFLoader,
    '.txt': TextLoader,
    '.docx': Docx2txtLoader,
  }

  documents = []
  for filename in os.listdir(DATA_PATH):
    ext = os.path.splitext(filename)[1]
    if ext in loaders:
      loader_class = loaders[ext]
      file_path = os.path.join(DATA_PATH, filename)
      print(f"Loading: {filename}")
      loader = loader_class(file_path)
      documents.extend(loader.load())

  print(f"Total documents loaded: {len(documents)}")
  return documents

# CHUNKING DOCUMENT
def split_documents(documents):
  print("Split the document into chunks...")
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
  )
  chunks = text_splitter.split_documents(documents)
  print(f"Generates {len(chunks)} pieces of text (chunks).")
  return chunks

# SAVE TO VECTOR DATABASE
def add_to_chroma(chunks):
  if not chunks:
    print("There are no documents to process.")
    return
  
  print("Saving to ChromaDB (Vector Store)...")

  # Initialize Chroma with local storage path
  db = Chroma.from_documents(
    chunks,
    get_embedding_function(),
    persist_directory=DB_PATH,
  )

  print(f"Success! The database is stored in {DB_PATH}")

def clear_database():
    """Reset the database using the ChromaDB API (without deleting files)"""
    try:
        # Connect to the existing database
        client = chromadb.PersistentClient(path=DB_PATH)
        
        # Get all collection names
        collections = client.list_collections()
        
        if not collections:
            print("Database is already empty.")
            return True
        
        # Delete each collection one by one
        for collection in collections:
            client.delete_collection(name=collection.name)
            print(f"Collection '{collection.name}' has been deleted.")
        
        print("All data has been deleted from the database.")
        return True
        
    except Exception as e:
        print(f"Error occurred while resetting the database: {e}")
        return False

# MAIN EXECUTION
if __name__ == "__main__":
  print("STARTING INGESTION PROCESS...")
  docs = load_documents()
  if docs:
    chunks = split_documents(docs)
    add_to_chroma(chunks)
  else:
    print("Place the PDF/DOCX/TXT file in the 'data/' folder, then restart.")