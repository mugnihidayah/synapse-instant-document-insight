"""
Ingestion module for document processing

This module provides:
- Document loaders (PDF, DOCX, TXT)
- Text Chunking
- Vector store management
"""

from src.ingestion.chunkers import create_text_splitter, split_documents
from src.ingestion.loaders import (
    get_supported_extensions,
    load_document_from_path,
    load_document_from_upload,
    load_documents_from_uploads,
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
