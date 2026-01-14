"""
Text chunking for document processing

Splits documents into smaller chunks for embedding and retrieval
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import settings


def create_text_splitter(
    chunk_size: int | None = None, chunk_overlap: int | None = None
) -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter with specified aprameters

    Args:
      chunk_size: Size of each chunk
      chunk_overlap: Overlap between chunks

    Returns:
      configured RecursiveCharacterTextSplitter
    """
    if chunk_size is None:
        chunk_size = settings.chunk_size

    if chunk_overlap is None:
        chunk_overlap = settings.chunk_overlap

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )


def split_documents(
    documents: list[Document], chunk_size: int | None = None, chunk_overlap: int | None = None
) -> list[Document]:
    """
    Split documents into smaller chunks

    Args:
      documents: List of langchain documents
      chunk_size: Size of each chunk
      chunk_overlap: Overlap between chunks

    Returns:
      List of chunked documents

    Example:
      docs = [Document(page_content="Long text...", metadata={})]
      chunks = split_documents(docs, chunk_size=500)
    """
    if not documents:
        return []

    splitter = create_text_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(documents)

    return chunks
