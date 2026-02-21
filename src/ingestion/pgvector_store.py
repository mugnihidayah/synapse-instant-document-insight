"""
PostgreSQL + pgvector vectorstore implementation.

Replace ChromaDB with persistent vector storage.
"""

import uuid
from typing import Any

from langchain_core.documents import Document as LangchainDocument
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.exceptions import VectorStoreError
from src.db.models import Document, Session
from src.ingestion.metadata_filters import build_metadata_filter_clause


def get_embedding_function() -> HuggingFaceEmbeddings:
    """Get the embedding function for vectorization."""
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


_embedding_fn: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Get cached embedding function."""
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = get_embedding_function()
    return _embedding_fn


async def store_documents(
    db: AsyncSession,
    session_id: uuid.UUID,
    documents: list[LangchainDocument],
    batch_size: int = 50,
) -> int:
    """
    Store documents with embeddings in PostgreSQL using batch processing.

    Processes documents in batches to reduce memory usage for large files.
    """
    import gc

    try:
        embeddings = get_embeddings()
        total_stored = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            texts = [doc.page_content for doc in batch]
            vectors = embeddings.embed_documents(texts)

            db_documents = []
            for doc, vector in zip(batch, vectors, strict=True):
                db_doc = Document(
                    session_id=session_id,
                    content=doc.page_content,
                    embedding=vector,
                    metadata_=doc.metadata or {},
                )
                db_documents.append(db_doc)

            db.add_all(db_documents)
            await db.flush()
            total_stored += len(db_documents)

            gc.collect()

        session = await db.get(Session, session_id)
        if session:
            session.document_count += total_stored

        return total_stored

    except Exception as exc:
        raise VectorStoreError(
            "Failed to store documents",
            details={"error": str(exc), "session_id": str(session_id)},
        ) from exc


async def similarity_search(
    db: AsyncSession,
    session_id: uuid.UUID,
    query: str,
    k: int = 5,
    filters: dict[str, Any] | None = None,
) -> list[LangchainDocument]:
    """Find similar documents using pgvector cosine similarity."""
    try:
        embeddings = get_embeddings()
        query_vector = embeddings.embed_query(query)

        filter_sql, filter_params = build_metadata_filter_clause(filters)

        stmt = text(
            f"""
            SELECT id, content, metadata, embedding <=> :query_vector AS distance
            FROM documents
            WHERE session_id = :session_id
            {filter_sql}
            ORDER BY embedding <=> :query_vector
            LIMIT :k
            """
        )

        params: dict[str, Any] = {
            "query_vector": str(query_vector),
            "session_id": str(session_id),
            "k": k,
            **filter_params,
        }
        result = await db.execute(stmt, params)

        rows = result.fetchall()

        documents = []
        for row in rows:
            doc = LangchainDocument(
                page_content=row.content,
                metadata={
                    **row.metadata,
                    "id": str(row.id),
                    "distance": row.distance,
                },
            )
            documents.append(doc)

        return documents

    except Exception as exc:
        raise VectorStoreError(
            "Failed to perform similarity search",
            details={"error": str(exc), "session_id": str(session_id)},
        ) from exc


async def delete_session_documents(
    db: AsyncSession,
    session_id: uuid.UUID,
) -> int:
    """Delete all documents for a session."""
    try:
        stmt = select(Document).where(Document.session_id == session_id)
        result = await db.execute(stmt)
        documents = result.scalars().all()

        count = len(documents)
        for doc in documents:
            await db.delete(doc)

        session = await db.get(Session, session_id)
        if session:
            session.document_count = 0

        return count

    except Exception as exc:
        raise VectorStoreError(
            "Failed to delete documents",
            details={"error": str(exc), "session_id": str(session_id)},
        ) from exc
