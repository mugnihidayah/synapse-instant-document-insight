"""
Ingestion module for document processing

This module provides:
- Document loaders (PDF, DOCX, TXT)
- Text Chunking
- Vector store management
"""

from src.ingestion.loaders import (
  load_document_from_path,
  load_document_from_upload,
  load_documents_from_uploads,
  get_supported_extensions,
)
from src.ingestion.chunkers import split_documents, create_text_splitter
from src.ingestion.vectorstore import (
  create_vectorstore,
  create_persistent_vectorstore,
  get_embedding_function,
)

__all__: list[str] = [
  "load_document_from_path",
  "load_document_from_upload",
  "load_documents_from_uploads",
  "get_supported_extensions",
  "split_documents",
  "create_text_splitter",
  "create_vectorstore",
  "create_persistent_vectorstore",
  "get_embedding_function",
]