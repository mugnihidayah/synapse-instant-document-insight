import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.documents import Document as LangchainDocument
from src.ingestion.pgvector_store import store_documents
@pytest.mark.asyncio
async def test_store_documents_success():
    """Test storing documents properly calls DB methods"""
    # Arrange
    mock_db = AsyncMock()
    # Mocking session get for updating count
    mock_session_record = MagicMock()
    mock_session_record.document_count = 0
    mock_db.get.return_value = mock_session_record
    session_id = uuid.uuid4()
    documents = [
        LangchainDocument(page_content="Test 1", metadata={"source": "test.pdf"}),
        LangchainDocument(page_content="Test 2", metadata={"source": "test.pdf"}),
    ]
    # Mock embedding function in the module
    with patch("src.ingestion.pgvector_store.get_embedding_function") as mock_get_embed:
        mock_embedding_service = MagicMock()
        # Mock 2 embeddings of dimension 384
        mock_embedding_service.embed_documents.return_value = [[0.1]*384, [0.2]*384]
        mock_get_embed.return_value = mock_embedding_service
        
        # Act
        count = await store_documents(mock_db, session_id, documents)
    # Assert
    assert count == 2
    assert mock_db.add_all.called
    assert mock_db.flush.called
    
    # Verify add_all was called with correct Document objects
    args, _ = mock_db.add_all.call_args
    added_docs = args[0]
    assert len(added_docs) == 2
    assert added_docs[0].session_id == session_id
    assert added_docs[0].content == "Test 1"