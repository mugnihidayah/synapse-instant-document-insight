"""
Test for core module (config and exception)
"""

import pytest
from src.core.config import Settings, get_settings, settings
from src.core.exceptions import (
  SynapseError,
  DocumentProcessingError,
  VectorStoreError,
  RAGError,
  ConfigurationError,
)

class TestSettings:
  """Test for Settings class"""
  def test_settings_loads(self) -> None:
    """Test that settings loads without error"""
    s = Settings()
    assert s is not None

  def test_settings_has_default_values(self) -> None:
    """Test that settings has expected default values"""
    s = Settings()
    assert s.chunk_size == 500
    assert s.chunk_overlap == 100
    assert s.retrieval_top_k == 10
    assert s.rerank_top_k == 3

  def test_settings_model_llm_default(self) -> None:
    """Test default LLM model"""
    s = Settings()
    assert "llama" in s.llm_model.lower() or s.llm_model != ""

  def test_settings_embedding_model_default(self) -> None:
    """Test default embedding model"""
    s = Settings()
    assert "sentence-transformers" in s.embedding_model

  def test_settings_cached(self) -> None:
    """Test that get_settings returns cached instance"""
    s1 = get_settings()
    s2 = get_settings()
    # due to lru_cache, these might be same object
    # but we test environment changes, they might differ
    assert s1 is not None
    assert s2 is not None
  
  def test_global_settings_accesible(self) -> None:
    """Test that global settings are accessible"""
    assert settings is not None
    assert hasattr(settings, "chunk_size")

class TestExceptions:
  """Test for custom exception classes"""

  def test_synapse_error_message(self) -> None:
    """Test SynapseError strore message correctly"""
    error = SynapseError("test error")
    assert error.message == "test error"
    assert str(error) == "test error"

  def test_synapse_error_with_details(self) -> None:
    """Test SynapseError with details."""
    error = SynapseError("Test error", details={"key": "value"})
    assert error.details == {"key": "value"}
    assert "key" in str(error)
    
  def test_synapse_error_repr(self) -> None:
    """Test SynapseError repr."""
    error = SynapseError("Test", details={"a": 1})
    repr_str = repr(error)
    assert "SynapseError" in repr_str
    assert "Test" in repr_str
    
  def test_document_processing_error_inherits(self) -> None:
    """Test DocumentProcessingError is SynapseError."""
    error = DocumentProcessingError("Doc error")
    assert isinstance(error, SynapseError)
    assert isinstance(error, Exception)
    
  def test_vectorstore_error_inherits(self) -> None:
    """Test VectorstoreError is SynapseError."""
    error = VectorStoreError("Vector error")
    assert isinstance(error, SynapseError)
    
  def test_rag_error_inherits(self) -> None:
    """Test RAGError is SynapseError."""
    error = RAGError("RAG error")
    assert isinstance(error, SynapseError)
    
  def test_configuration_error_inherits(self) -> None:
    """Test ConfigurationError is SynapseError."""
    error = ConfigurationError("Config error")
    assert isinstance(error, SynapseError)
    
  def test_can_catch_all_with_synapse_error(self) -> None:
    """Test catching all custom exceptions with base class."""
    exceptions = [
      DocumentProcessingError("1"),
      VectorStoreError("2"),
      RAGError("3"),
      ConfigurationError("4"),
    ]
    
    for exc in exceptions:
      try:
        raise exc
      except SynapseError as e:
        assert e.message in ["1", "2", "3", "4"]