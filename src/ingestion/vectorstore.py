"""
Vector store for management using ChromaDB

Provides in memory and persistent vertor store operations
"""

import uuid

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.core.config import settings
from src.core.exceptions import VectorStoreError


def get_embedding_function() -> HuggingFaceEmbeddings:
  """
  Get embedding function using Huggingface model

  Returns:
    HuggingFaceEmbeddings instance
  """
  return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def create_vectorstore(documents: list[Document], collection_name: str | None = None) -> Chroma:
  """
  Create an in memory vector store from documents

  Args:
    documents: List of langchain documents to store
    collection_name: Optional collection name

  Returns:
    Chroma instance of vector store

  Raises:
    VectorStoreError: If creation fails
  """
  if not documents:
    raise VectorStoreError(
      "Cannot create vector store from empty documents", details={"num_documents": 0}
    )

  if collection_name is None:
    collection_name = f"session_{uuid.uuid4().hex[:8]}"

  try:
    # create in memory client
    client = chromadb.EphemeralClient()

    vectorstore = Chroma.from_documents(
      documents=documents,
      embedding=get_embedding_function(),
      client=client,
      collection_name=collection_name,
    )

    return vectorstore

  except Exception as e:
    raise VectorStoreError(
      "Failed to create vectorstore",
      details={"error": str(e), "num_documents": len(documents)},
    ) from e


def create_persistent_vectorstore(
  documents: list[Document], persist_directory: str, collection_name: str = "documents"
) -> Chroma:
  """
  Create a persistent vector store that saves to disk

  Args:
    documents: List of langchain documents to store
    persist_directory: Directory to save vector store
    collection_name: Optional collection name

  Returns:
    Chroma instance of vector store

  Raises:
    VectorStoreError: If creation fails
  """
  if not documents:
    raise VectorStoreError(
      "Cannot create vector store from empty documents", details={"num_documents": 0}
    )

  try:
    vectorstore = Chroma.from_documents(
      documents=documents,
      embedding=get_embedding_function(),
      persist_directory=persist_directory,
      collection_name=collection_name,
    )

    return vectorstore

  except Exception as e:
    raise VectorStoreError(
      "Failed to create persistent vectorstore",
      details={"error": str(e), "directory": persist_directory},
    ) from e
