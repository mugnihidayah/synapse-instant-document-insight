"""
Pytest configuration and shared fixtures

Fixtures defined here are automatically available to all test
"""

import os
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document


# Environment Fixtures
@pytest.fixture(autouse=True)
def setup_test_environment() -> Generator[None, None, None]:
    """
    Setup test environment before each test

    This fixture runs automatically for all tests
    It sets dummy API keys to prevent errors during testing
    """
    original_env = os.environ.copy()

    os.environ["GROQ_API_KEY"] = "test_groq_key_12345"
    os.environ["HUGGINGFACE_TOKEN"] = "test_hf_token_12345"

    yield

    # restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# Document Fixtures
@pytest.fixture
def sample_document() -> Document:
    """
    Create a single sample document for testing
    """
    return Document(
        page_content="This is a sample document for testing purposes",
        metadata={"source": "test.txt", "page": 1},
    )


@pytest.fixture
def sample_documents() -> list[Document]:
    """
    Create multiple sample documents for testing
    """
    return [
        Document(
            page_content="Python is a programming language",
            metadata={"source": "doc1.txt", "page": 1},
        ),
        Document(
            page_content="Machine learning is a subset of AI",
            metadata={"source": "doc2.txt", "page": 1},
        ),
        Document(
            page_content="RAG combines retrieval with generation",
            metadata={"source": "doc3.txt", "page": 1},
        ),
    ]


@pytest.fixture
def long_document() -> Document:
    """Create a long document for chunking tests"""
    # Create text longer than default chunk size
    paragraphs = ["This is paragraph {i}. " * 20 for i in range(10)]
    content = "\n\n".join(paragraphs)

    return Document(page_content=content, metadata={"source": "long_doc.txt", "page": 1})


# File Fixtures
@pytest.fixture
def temp_text_file() -> Generator[Path, None, None]:
    """Create a temporary text file for testing"""

    content = "This is test content. \nLine 2. \nLine 3."

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

        yield temp_path

        # cleanup
        if temp_path.exists():
            temp_path.unlink()


@pytest.fixture
def temp_directory() -> Generator[Path, None, None]:
    """Create a temporary directory for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


# Mock Fixtures
@pytest.fixture
def mock_chat_messages() -> list[dict]:
    """Create mock chat history for testing"""
    return [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
        {"role": "user", "content": "Tell me more."},
    ]


@pytest.fixture
def mock_rerank_passages() -> list[dict]:
    """Create mock passages for reranking testing"""
    return [
        {"id": "1", "text": "Python is great for ML.", "meta": {}},
        {"id": "2", "text": "Java is used for enterprise.", "meta": {}},
        {"id": "3", "text": "Python has many libraries.", "meta": {}},
    ]


@pytest.fixture
def mock_vectorstore() -> MagicMock:
    """Create a mock vectorstore for testing."""
    from langchain_core.documents import Document

    mock = MagicMock()
    mock_retriever = MagicMock()
    mock.as_retriever.return_value = mock_retriever
    mock_retriever.invoke.return_value = [Document(page_content="Test content", metadata={})]

    return mock
