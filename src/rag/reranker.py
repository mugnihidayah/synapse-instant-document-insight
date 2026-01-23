"""
Reranker module for improving retrieval quality

Uses Flashrank for neural reranking of retrieved documents
"""

from functools import lru_cache

from flashrank import Ranker, RerankRequest

from src.core.config import settings


@lru_cache(maxsize=1)
def get_reranker() -> Ranker:
    """
    Get cache reranker instance

    Uses lru_cache to avoid loading model multiple times

    Returns:
      Ranker: Flashrank Reranker instance
    """
    return Ranker(model_name=settings.reranker_model)


def rerank_documents(query: str, documents: list[dict], top_k: int | None = None) -> list[dict]:
    """
    Rerank documents based on query

    Args:
      query: Search query
      documents: List of documents with 'id', 'text', and 'meta' keys
      top_k: Number of documents to return

    Returns:
      List of reranked documents sorted by relevance to query

    Example:
      documents = [
        {"id": "1", "text": "Python is a programming language", "meta": {}},
        {"id": "2", "text": "Java is also a programming language", "meta": {}},
      ]
      results = rerank_documents("What is Python?", documents, top_k=1)
    """

    if not documents:
        return []

    if top_k is None:
        top_k = settings.rerank_top_k

    reranker = get_reranker()

    # create rerank request
    request = RerankRequest(query=query, passages=documents)

    # get reranked results
    results: list[dict] = reranker.rerank(request)

    return results[:top_k]
