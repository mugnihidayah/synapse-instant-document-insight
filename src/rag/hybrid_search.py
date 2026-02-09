"""
Hybrid search combining vector and keyword search.

Uses Reciprocal Rank Fusion (RRF) to merge results.
"""

import uuid

from langchain_core.documents import Document as LangchainDocument
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logger import get_logger
from src.ingestion.pgvector_store import similarity_search

logger = get_logger(__name__)


async def keyword_search(
    db: AsyncSession,
    session_id: uuid.UUID,
    query: str,
    k: int = 10,
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
    # convert query to tsquery format
    # replace spaces with AND operator for phrase matching
    tsquery = " & ".join(query.split())

    stmt = text("""
        SELECT id, content, metadata,
            ts_rank(content_tsv, to_tsquery(:query)) AS rank
        FROM documents
        WHERE session_id = :session_id
            AND content_tsv @@ to_tsquery(:query)
        ORDER BY rank DESC
        LIMIT :k
    """)

    try:
        result = await db.execute(
            stmt,
            {
                "session_id": str(session_id),
                "query": tsquery,
                "k": k,
            },
        )
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
    # run both searches
    vector_docs = await similarity_search(db, session_id, query, k=k * 2)
    keyword_docs = await keyword_search(db, session_id, query, k=k * 2)

    # calculate RRF scores
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, LangchainDocument] = {}

    # RRF constant
    rrf_k = 60

    # score vector results
    for rank, doc in enumerate(vector_docs):
        doc_id = doc.metadata.get("id", str(rank))
        score = vector_weight * (1 / (rrf_k + rank + 1))
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + score
        doc_map[doc_id] = doc

    # score keyword results
    for rank, doc in enumerate(keyword_docs):
        doc_id = doc.metadata.get("id", str(rank))
        score = keyword_weight * (1 / (rrf_k + rank + 1))
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + score
        doc_map[doc_id] = doc

    # sort by RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    # build result list
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
