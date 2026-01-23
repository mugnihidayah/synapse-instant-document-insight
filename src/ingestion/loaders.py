"""
Document loaders for varius formats

Supports PDF, DOCX, and TXT files
"""

import os
import tempfile
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import (
  Docx2txtLoader,
  PyMuPDFLoader,
  TextLoader,
)
from langchain_core.documents import Document

from src.core.exceptions import DocumentProcessingError

# mapping of file extensions to loader classes
LOADER_MAPPING: dict[str, type] = {
  ".pdf": PyMuPDFLoader,
  ".docx": Docx2txtLoader,
  ".txt": TextLoader,
}


def get_supported_extensions() -> list[str]:
  """
  Get list of supported file extensions

  Returns:
    List of extension strings
  """
  return list(LOADER_MAPPING.keys())


def load_document_from_path(file_path: str | Path) -> list[Document]:
  """
  Load document from file path

  Args:
    file_path: Path to document file

  Returns:
    List of langchain documents

  Raises:
    DocumentProcessingError: If file extension is not supported or loading fails
  """

  file_path = Path(file_path)

  if not file_path.exists():
    raise DocumentProcessingError("File not found", details={"path": str(file_path)})

  ext = file_path.suffix.lower()
  if ext not in LOADER_MAPPING:
    raise DocumentProcessingError(
      "Unsupported file format",
      details={"extension": ext, "supported_extensions": get_supported_extensions()},
    )

  loader_class = LOADER_MAPPING[ext]
  loader = loader_class(str(file_path))

  try:
    documents: list[Document] = loader.load()
  except Exception as e:
    raise DocumentProcessingError(
      "Failed to load document", details={"path": str(file_path), "error": str(e)}
    ) from e

  return documents


def load_document_from_upload(uploaded_file: Any, filename: str) -> list[Document]:
  """
  Load document from uploaded file

  Args:
    uploaded_file: file like object
    filename: original filename

  Returns:
    List of langchain documents

  Raises:
    DocumentProcessingError: If file extension is not supported or loading fails
  """

  ext = os.path.splitext(filename)[1].lower()

  if ext not in LOADER_MAPPING:
    raise DocumentProcessingError(
      "Unsupported file format",
      details={"filename": filename, "supported": get_supported_extensions()},
    )

  # save to temp file for processing
  with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
    tmp.write(uploaded_file.read())
    tmp_path = tmp.name

  try:
    loader_class = LOADER_MAPPING[ext]
    loader = loader_class(tmp_path)
    documents: list[Document] = loader.load()

    # update metadata with original filename
    for doc in documents:
      doc.metadata["source"] = filename
      if "page" in doc.metadata:
        doc.metadata["page"] = doc.metadata["page"] + 1

    return documents
  except Exception as e:
    raise DocumentProcessingError(
      "Failed to load uploaded document", details={"filename": filename, "error": str(e)}
    ) from e

  finally:
    # clean up temp file
    os.unlink(tmp_path)


def load_documents_from_uploads(uploaded_files: list) -> list[Document]:
  """
  Load multiple documents from uploaded files

  Args:
    uploaded_files: list of uploaded files

  Returns:
    Combined list of langchain documents
  """
  all_documents: list[Document] = []

  for uploaded_file in uploaded_files:
    docs = load_document_from_upload(uploaded_file, uploaded_file.name)
    all_documents.extend(docs)

  return all_documents
