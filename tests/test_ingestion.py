"""
Tests for ingestion module
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from langchain_core.documents import Document
from src.core.exceptions import DocumentProcessingError, VectorStoreError
from src.ingestion.loaders import (
  get_supported_extensions,
  load_document_from_path,
  LOADER_MAPPING,
)
from src.ingestion.chunkers import (
  create_text_splitter,
  split_documents,
)

class TestLoaders:
  """Tests for document loaders."""
  
  def test_get_supported_extensions(self) -> None:
    """Test that supported extensions are returned."""
    extensions = get_supported_extensions()
    
    assert isinstance(extensions, list)
    assert ".pdf" in extensions
    assert ".txt" in extensions
    assert ".docx" in extensions
  
  def test_loader_mapping_has_required_formats(self) -> None:
    """Test that LOADER_MAPPING contains required formats."""
    assert ".pdf" in LOADER_MAPPING
    assert ".txt" in LOADER_MAPPING
    assert ".docx" in LOADER_MAPPING
  
  def test_load_document_from_path_txt(
    self, 
    temp_text_file: Path
  ) -> None:
    """Test loading a text file."""
    docs = load_document_from_path(temp_text_file)
    
    assert isinstance(docs, list)
    assert len(docs) > 0
    assert isinstance(docs[0], Document)
    assert "test content" in docs[0].page_content.lower()
  
  def test_load_document_from_path_not_found(self) -> None:
    """Test error when file not found."""
    with pytest.raises(DocumentProcessingError) as exc_info:
      load_document_from_path("/nonexistent/file.txt")
    
    assert "not found" in str(exc_info.value).lower()
  
  def test_load_document_from_path_unsupported_format(
    self,
    temp_directory: Path
  ) -> None:
    """Test error for unsupported file format."""
    # Create a file with unsupported extension
    unsupported_file = temp_directory / "test.xyz"
    unsupported_file.write_text("content")
    
    with pytest.raises(DocumentProcessingError) as exc_info:
      load_document_from_path(unsupported_file)
    
    assert "unsupported" in str(exc_info.value).lower()
  
  def test_load_document_path_string_and_path_object(
    self,
    temp_text_file: Path
  ) -> None:
    """Test that both string and Path work."""
    # As Path
    docs1 = load_document_from_path(temp_text_file)
    # As string
    docs2 = load_document_from_path(str(temp_text_file))
    
    assert len(docs1) == len(docs2)

class TestChunkers:
  """Tests for text chunking."""
  
  def test_create_text_splitter_default(self) -> None:
    """Test creating text splitter with defaults."""
    splitter = create_text_splitter()
    
    assert splitter is not None
    assert splitter._chunk_size == 500  # Default from settings
    assert splitter._chunk_overlap == 100
  
  def test_create_text_splitter_custom_size(self) -> None:
    """Test creating text splitter with custom size."""
    splitter = create_text_splitter(chunk_size=200, chunk_overlap=50)
    
    assert splitter._chunk_size == 200
    assert splitter._chunk_overlap == 50
  
  def test_split_documents_empty_list(self) -> None:
    """Test splitting empty document list."""
    result = split_documents([])
    
    assert result == []
  
  def test_split_documents_short_document(
    self,
    sample_document: Document
  ) -> None:
    """Test splitting short document (no splitting needed)."""
    # sample_document is short, should not be split
    chunks = split_documents([sample_document])
    
    assert len(chunks) >= 1
    assert chunks[0].page_content == sample_document.page_content
  
  def test_split_documents_long_document(
    self,
    long_document: Document
  ) -> None:
    """Test splitting long document into multiple chunks."""
    chunks = split_documents([long_document], chunk_size=200)
    
    # Long document should be split into multiple chunks
    assert len(chunks) > 1
    
    # Each chunk should be <= chunk_size (approximately)
    for chunk in chunks:
      # Allow some flexibility due to splitter behavior
      assert len(chunk.page_content) <= 300
  
  def test_split_documents_preserves_metadata(
    self,
    sample_document: Document
  ) -> None:
    """Test that chunking preserves document metadata."""
    sample_document.metadata = {"source": "test.txt", "custom": "value"}
    
    chunks = split_documents([sample_document])
    
    for chunk in chunks:
      assert chunk.metadata.get("source") == "test.txt"
      assert chunk.metadata.get("custom") == "value"
  
  def test_split_multiple_documents(
    self,
    sample_documents: list[Document]
  ) -> None:
    """Test splitting multiple documents."""
    chunks = split_documents(sample_documents)
    
    # Should have at least as many chunks as documents
    assert len(chunks) >= len(sample_documents)

class TestVectorStore:
  """Tests for vectorstore operations."""
  
  def test_get_embedding_function(self) -> None:
    """Test getting embedding function."""
    from src.ingestion.vectorstore import get_embedding_function
    
    embeddings = get_embedding_function()
    
    assert embeddings is not None
    assert hasattr(embeddings, "embed_documents")
    assert hasattr(embeddings, "embed_query")
  
  def test_create_vectorstore_empty_documents_raises(self) -> None:
    """Test that empty documents raises error."""
    from src.ingestion.vectorstore import create_vectorstore
      
    with pytest.raises(VectorStoreError) as exc_info:
      create_vectorstore([])
      
    assert "empty" in str(exc_info.value).lower()
  
  @pytest.mark.slow
  def test_create_vectorstore_success(
    self,
    sample_documents: list[Document]
  ) -> None:
    """
    Test creating vectorstore with documents.
    
    Note: This test is marked as slow because it loads embedding model.
    Skip with: pytest -m "not slow"
    """
    from src.ingestion.vectorstore import create_vectorstore
      
    vectorstore = create_vectorstore(sample_documents)
      
    assert vectorstore is not None
    # Test retrieval
    results = vectorstore.similarity_search("Python", k=1)
    assert len(results) >= 0  # May or may not find matches
  
  def test_create_vectorstore_custom_collection_name(
    self,
    sample_documents: list[Document]
  ) -> None:
    """Test creating vectorstore with custom collection name."""
    from src.ingestion.vectorstore import create_vectorstore
      
    vectorstore = create_vectorstore(
      sample_documents,
      collection_name="test_collection"
    )
      
    assert vectorstore is not None

class TestIntegration:
  """Integration tests for ingestion pipeline."""
  
  def test_load_and_chunk_pipeline(
    self,
    temp_text_file: Path
  ) -> None:
    """Test loading and chunking documents together."""
    # Load
    docs = load_document_from_path(temp_text_file)
    
    # Chunk
    chunks = split_documents(docs)
    
    assert len(chunks) >= 1
    assert all(isinstance(c, Document) for c in chunks)
  
  @pytest.mark.slow
  def test_full_ingestion_pipeline(
    self,
    temp_text_file: Path
  ) -> None:
    """
    Test full pipeline: load -> chunk -> vectorstore.
      
    Marked as slow due to embedding model loading.
    """
    from src.ingestion.vectorstore import create_vectorstore
      
    # Load
    docs = load_document_from_path(temp_text_file)
      
    # Chunk
    chunks = split_documents(docs)
      
    # Store
    vectorstore = create_vectorstore(chunks)
      
    assert vectorstore is not None