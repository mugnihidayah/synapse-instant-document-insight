"""
Reranker module with switchable implementations

Supports:
- Cohere Rerank API
- Local Cross-encoder
"""

from pydantic_settings import SettingsConfigDict
from transformers.models.yoso.modeling_yoso import lsh_cumulation
from abc import ABC, abstractmethod

from langchain_core.documents import Document

from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger(__name__)


class BaseReranker(ABC):
    """Abstract base class for reranker"""

    @abstractmethod
    async def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        """Rerank documents bases on relevance to query"""
        pass


class CohereReranker(BaseReranker):
    """Reranker using Cohere Rerank API"""

    def __init__(self):
        import cohere

        self.client = cohere.AsyncClient(api_key=settings.cohere_api_key)
        self.model = "rerank-english-v3.0"
        logger.info("Cohere reranker initialized", model=self.model)

    async def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        """Rerank using Cohere API"""
        if not documents:
            return []

        if len(documents) <= top_k:
            return documents

        try:
            # extract text for reranking
            texts = [doc.page_content for doc in documents]

            # call cohere rerank API
            response = await self.client.rerank(
                query=query,
                documents=texts,
                top_n=top_k,
                model=self.model,
            )

            # reorder documents based on results
            reranked_docs = []
            for result in response.results:
                doc = documents[result.index]
                doc.metadata["rerank_score"] = result.relevance_score
                reranked_docs.append(doc)

            logger.info(
                "Cohere Rerank Completed",
                input_count=len(documents),
                output_count=len(reranked_docs),
            )

            return reranked_docs

        except Exception as e:
            logger.error("Cohere Rerank Failed", error=str(e))
            # fallback return original top_k
            return documents[:top_k]


class LocalReranker(BaseReranker):
    """Reranker using local model"""

    def __init__(self):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512,
        )
        logger.info("Local Reranker initialized", model="ms-marco-MiniLM-L-6-v2")

    async def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        """Rerank using local model"""
        if not documents:
            return []

        if len(documents) <= top_k:
            return documents

        try:
            # prepare query document pairs
            pairs = [[query, doc.page_content] for doc in documents]

            # get scores from cross-encoder
            scores = self.model.predict(pairs)

            # sort by score (descending)
            scored_docs = list(zip(documents, scores, strict=True))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # return top_k with scores
            reranked_docs = []
            for doc, score in scored_docs[:top_k]:
                doc.metadata["rerank_score"] = float(score)
                reranked_docs.append(doc)

            logger.info(
                "Local Rerank Completed",
                input_count=len(documents),
                output_count=len(reranked_docs),
            )

            return reranked_docs

        except Exception as e:
            logger.error("Local Rerank Failed", error=str(e))
            return documents[:top_k]


class NoOpReranker(BaseReranker):
    """No operation reranker that returns without reranking"""

    async def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        """Return documents without reranking"""
        return documents[:top_k]


# cache for reranker
_reranker: BaseReranker | None = None


def get_reranker() -> BaseReranker:
    """
    Get reranker instance
    """

    global _reranker

    if _reranker is not None:
        return _reranker

    provider = settings.reranker_provider.lower()

    if provider == "cohere":
        if not settings.cohere_api_key:
            logger.warning("Cohere API key not found, using NoOpReranker")
            _reranker = NoOpReranker()
        else:
            _reranker = CohereReranker()
    elif provider == "local":
        _reranker = LocalReranker()
    else:
        logger.info("Reranker Disabled", provider=provider)
        _reranker = NoOpReranker()

    return _reranker
