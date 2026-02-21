"""
Text chunking for document processing.

Splits documents into smaller chunks for embedding and retrieval.
"""

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import settings

SECTION_PATTERN = re.compile(r"^(?:#{1,6}\s+|\d+(?:\.\d+)*\s+)(.+)$")


def _adaptive_chunk_params(documents: list[Document]) -> tuple[int, int]:
    """Adjust chunk size/overlap by average source length."""
    if not documents:
        return settings.chunk_size, settings.chunk_overlap

    avg_len = sum(len(doc.page_content) for doc in documents) / len(documents)

    if avg_len < 1200:
        return max(500, settings.chunk_size - 250), max(80, settings.chunk_overlap - 80)
    if avg_len > 5000:
        return min(1600, settings.chunk_size + 300), min(350, settings.chunk_overlap + 80)
    return settings.chunk_size, settings.chunk_overlap


def _extract_section_heading(text: str) -> str | None:
    for raw_line in text.splitlines()[:8]:
        line = raw_line.strip()
        if not line:
            continue
        match = SECTION_PATTERN.match(line)
        if match:
            return match.group(1).strip()
        if line.isupper() and 5 <= len(line) <= 120:
            return line.title()
    return None


def create_text_splitter(
    chunk_size: int | None = None, chunk_overlap: int | None = None
) -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter with specified parameters.

    Args:
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        Configured RecursiveCharacterTextSplitter
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n## ", "\n# ", "\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )


def split_documents(
    documents: list[Document], chunk_size: int | None = None, chunk_overlap: int | None = None
) -> list[Document]:
    """
    Split documents into smaller chunks.

    Adds stable metadata fields for downstream filtering and citations.
    """
    if not documents:
        return []

    adaptive_size, adaptive_overlap = _adaptive_chunk_params(documents)
    splitter = create_text_splitter(
        chunk_size=chunk_size or adaptive_size,
        chunk_overlap=chunk_overlap or adaptive_overlap,
    )

    chunks: list[Document] = []

    for doc in documents:
        per_doc_chunks = splitter.split_documents([doc])

        for index, chunk in enumerate(per_doc_chunks):
            metadata = dict(chunk.metadata)
            metadata.setdefault("chunk_type", "content")
            metadata["chunk_index"] = index
            metadata["chunk_size"] = len(chunk.page_content)

            section = _extract_section_heading(chunk.page_content)
            if section:
                metadata["section"] = section

            chunks.append(Document(page_content=chunk.page_content, metadata=metadata))

    return chunks


def extract_document_header(documents: list[Document]) -> Document | None:
    """
    Extract document header as a metadata chunk.

    Args:
        documents: List of LangChain documents from loader.

    Returns:
        Document with first-page content and metadata.
    """
    if not documents:
        return None

    first_page = documents[0]
    header_content = first_page.page_content[:2000]

    metadata = dict(first_page.metadata)
    metadata["chunk_type"] = "document_header"
    metadata["chunk_size"] = len(header_content)

    section = _extract_section_heading(header_content)
    if section:
        metadata["section"] = section

    return Document(
        page_content=header_content,
        metadata=metadata,
    )
