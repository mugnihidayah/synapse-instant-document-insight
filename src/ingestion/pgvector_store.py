"""
PostgreSQL + pgvector vectorstore implementation
Replace ChromaDB with persistent vector storage
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

def get_embedding_function() -> HuggingFaceEmbeddings:
  """Get the embedding function for vectorization"""
  return HuggingFaceEmbeddings(
    model_name=settings.embedding_model,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
  )

# global embedding function (cached)
_embedding_fn: HuggingFaceEmbeddings | None = None

def get_embeddings() -> HuggingFaceEmbeddings:
  """Get cached embedding function"""
  global _embedding_fn
  if _embedding_fn is None:
    _embedding_fn = get_embedding_function()
  return _embedding_fn

async def store_documents(
  db: AsyncSession,
  session_id: uuid.UUID,
  documents: list[LangchainDocument],
) -> int:
  """
  Store documents with embeddings in PostgreSQL

  Args:
    db: Database session
    session_id: Session ID
    documents: List of Langchain documents

  Returns:
    Number of documents stored
  """

  try:
    embeddings = get_embeddings()

    # get embeddings for all documents
    texts = [doc.page_content for doc in documents]
    vectors = embeddings.embed_documents(texts)

    # create document records
    db_documents = []
    for doc, vector in zip(documents, vectors):
      db_doc = Document(
        session_id=session_id,
        content=doc.page_content,
        embedding=vector,
        metadata=doc.metadata or {}
      )
      db_documents.append(db_doc)

    db.add_all(db_documents)
    await db.flush()

    # update session document count
    session = await db.get(Session, session_id)
    if session:
      session.document_count += len(db_documents)

    return len(db_documents)

  except Exception as e:
    raise VectorStoreError(
      "Failed to store documents",
      details={"error": str(e), "session_id": str(session_id)},
    ) from e

async def similarity_search(
  db: AsyncSession,
  session_id: uuid.UUID,
  query: str,
  k: int = 5,
) -> list[LangchainDocument]:
  """
  Find similar documents using pgvector cosine similarity

  Args:
    db: Database session
    session_id: Session ID
    query: Query text
    k: Number of results to return

  Returns:
    List of similar Langchain documents
  """

  try:
    embeddings = get_embeddings()

    # get query embedding
    query_vector = embeddings.embed_query(query)

    stmt = text("""
      SELECT id, content, metadata, embedding <=> :query_vector AS distance
      FROM documents
      WHERE session_id = :session_id
      ORDER BY embedding <=> :query_vector
      LIMIT :k
    """)

    result = await db.execute(
      stmt,
      {
        "query_vector": str(query_vector),
        "session_id": str(session_id),
        "k": k,
      },
    )

    rows = result.fetchall()

    # convert to langchain documents
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

  except Exception as e:
    raise VectorStoreError(
      "Failed to perform similarity search",
      details={"error": str(e), "session_id": str(session_id)},
    ) from e

async def delete_session_documents(
  db: AsyncSession,
  session_id: uuid.UUID,
) -> int:
  """
  Delete all documents for a session

  Args:
    db: Database session
    session_id: Session ID

  Returns:
    Number of documents deleted
  """

  try:
    stmt = select(Document).where(Document.session_id == session_id)
    result = await db.execute(stmt)
    documents = result.scalars().all()

    count = len(documents)
    for doc in documents:
      await db.delete(doc)

    # update session count
    session = await db.get(Session, session_id)
    if session:
      session.document_count = 0

    return count

  except Exception as e:
    raise VectorStoreError(
      "Failed to delete documents",
      details={"error": str(e), "session_id": str(session_id)},
    ) from e