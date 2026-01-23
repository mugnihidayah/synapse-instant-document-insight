"""
Custom exception classes for Synapse RAG

This module defines a hierarchy or exceptions for better error handling:
- SynapseError: Base exception for all synapse related errors
- DocumentProcessingError: Errors during document loading / processing
- VectorstoreError: Errors with vector database
- RAGError: Errors during RAG execution

Usage:
  from src.core.exceptions import DocumentProcessingError

  try:
    process_document(file)
  except DocumentProcessingError as e:
    logger.error(f"Failed to process document: {e}")
"""


class SynapseError(Exception):
  """
  Base exception for all Synapse RAG errors

  All custom exceptions should inherit from this class to allow
  catching all synapse related errors.

  Attributes:
    message: Human readable error message
    details: Optional dictionary with additional error context
  """

  def __init__(self, message: str, details: dict | None = None) -> None:
    super().__init__(message)
    self.message: str = message
    self.details: dict = details or {}

  def __str__(self) -> str:
    if self.details:
      return f"{self.message} | Details: {self.details}"
    return self.message

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}(message={self.message!r}, details={self.details!r})"


class DocumentProcessingError(SynapseError):
  """
  Exception raised when document processing fails

  This can occur during:
  - File loading (unsupported format, corrupted file)
  - Text extraction (encoding issues, empty content)
  - Chunking (invalid parameters)
  """

  pass


class VectorStoreError(SynapseError):
  """
  Exception raised when vector store operation fail

  This can occur during:
  - Creating embeddings
  - Storing documents in ChromaDB
  - Retrieving documents
  """

  pass


class RAGError(SynapseError):
  """
  Exception raised when RAG execution fails

  This can occur during:
  - LLM API calls (rate limits, invalid key)
  - Reranking
  - Response generation
  """

  pass


class ConfigurationError(SynapseError):
  """
  Exception raised when configuration is invalid or missing

  This can occur during:
  - Required API keys are missing
  - Invalid configuration values
  - Environment setup fails
  """

  pass
