"""
Hybrid search combining vector and keyword search.

Uses Reciprocal Rank Fusion (RRF) to merge results.
"""

import uuid
from typing import Any

from langchain_core.documents import Document as LangchainDocument
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logger import get_logger
from src.ingestion.metadata_filters import build_metadata_filter_clause
from src.ingestion.pgvector_store import similarity_search

logger = get_logger(__name__)


async def keyword_search(
    db: AsyncSession,
    session_id: uuid.UUID,
    query: str,
    k: int = 10,
    filters: dict[str, Any] | None = None,
) -> list[LangchainDocument]:
    """
    Perform keyword search using PostgreSQL full-text search.

    Args:
        db: Database session
        sesssion_id: Session ID
        query: Search query
        k: Number of results

    Returns:
        List of matching documents
    """
    filter_sql, filter_params = build_metadata_filter_clause(filters, param_prefix="kw_")

    stmt = text(
        f"""
        SELECT id, content, metadata,
            ts_rank(content_tsv, plainto_tsquery('simple', :query)) AS rank
        FROM documents
        WHERE session_id = :session_id
            AND content_tsv @@ plainto_tsquery('simple', :query)
            {filter_sql}
        ORDER BY rank DESC
        LIMIT :k
        """
    )

    try:
        params: dict[str, Any] = {
            "session_id": str(session_id),
            "query": query,
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
                    "keyword_rank": float(row.rank),
                },
            )
            documents.append(doc)

        logger.debug("keyword_search_completed", count=len(documents))
        return documents

    except Exception as e:
        logger.warning("keyword_search_failed", error=str(e))
        return []


async def hybrid_search(
    db: AsyncSession,
    session_id: uuid.UUID,
    query: str,
    k: int = 10,
    vector_weight: float = 0.5,
    keyword_weight: float = 0.5,
    filters: dict[str, Any] | None = None,
) -> list[LangchainDocument]:
    """
    Perform hybrid search combining vector and keyword search.

    Uses Reciprocal Rank Fusion (RRF) to merge results.

    Args:
        db: Database session
        session_id: Session ID
        query: Search query
        k: Number of results
        vector_weight: Weight for vector search (0-1)
        keyword_weight: Weight for keyword search (0-1)

    Returns:
        List of documents sorted by hybrid score
    """
    vector_docs = await similarity_search(db, session_id, query, k=k * 2, filters=filters)
    keyword_docs = await keyword_search(db, session_id, query, k=k * 2, filters=filters)

    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, LangchainDocument] = {}

    rrf_k = 60

    for rank, doc in enumerate(vector_docs):
        doc_id = doc.metadata.get("id", str(rank))
        score = vector_weight * (1 / (rrf_k + rank + 1))
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + score
        doc_map[doc_id] = doc

    for rank, doc in enumerate(keyword_docs):
        doc_id = doc.metadata.get("id", str(rank))
        score = keyword_weight * (1 / (rrf_k + rank + 1))
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + score
        doc_map[doc_id] = doc

    sorted_ids = sorted(rrf_scores.keys(), key=lambda item: rrf_scores[item], reverse=True)

    results = []
    for doc_id in sorted_ids[:k]:
        doc = doc_map[doc_id]
        doc.metadata["hybrid_score"] = rrf_scores[doc_id]
        results.append(doc)

    logger.info(
        "hybrid_search_completed",
        vector_count=len(vector_docs),
        keyword_count=len(keyword_docs),
        final_count=len(results),
    )

    return results
