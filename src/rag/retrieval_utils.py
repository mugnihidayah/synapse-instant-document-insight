"""Utilities for retrieval-time query tuning and citation formatting."""

import math
import re
from collections.abc import Iterable

from langchain_core.documents import Document

from src.core.config import settings

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]{2,}")
STOPWORDS = {
    "the",
    "is",
    "are",
    "was",
    "were",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "for",
    "on",
    "at",
    "with",
    "yang",
    "dan",
    "atau",
    "di",
    "ke",
    "dari",
    "untuk",
    "apa",
    "itu",
    "ini",
    "pada",
}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def compute_dynamic_top_k(question: str, requested_top_k: int | None = None) -> int:
    """Compute retrieval depth from query complexity unless explicitly provided."""
    if requested_top_k is not None:
        return requested_top_k

    tokens = [token for token in tokenize(question) if token not in STOPWORDS]
    complexity = len(tokens)

    if complexity <= 4:
        target = settings.dynamic_top_k_min
    elif complexity <= 10:
        target = settings.dynamic_top_k_min + math.ceil((complexity - 4) / 2)
    else:
        target = settings.dynamic_top_k_max

    return max(settings.dynamic_top_k_min, min(target, settings.dynamic_top_k_max))


def _doc_relevance(query_tokens: set[str], doc: Document) -> float:
    rerank_score = doc.metadata.get("rerank_score")
    if isinstance(rerank_score, (int, float)):
        return float(rerank_score)

    hybrid_score = doc.metadata.get("hybrid_score")
    if isinstance(hybrid_score, (int, float)):
        return float(hybrid_score)

    distance = doc.metadata.get("distance")
    if isinstance(distance, (int, float)):
        return max(0.0, 1.0 - float(distance))

    doc_tokens = set(tokenize(doc.page_content))
    if not doc_tokens:
        return 0.0
    overlap = len(query_tokens & doc_tokens)
    return overlap / max(1, len(query_tokens))


def _lexical_similarity(text_a: str, text_b: str) -> float:
    tokens_a = set(tokenize(text_a))
    tokens_b = set(tokenize(text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def apply_mmr_diversification(
    documents: list[Document],
    query: str,
    top_k: int,
    lambda_mult: float,
) -> list[Document]:
    """Select diverse but relevant documents with lightweight lexical MMR."""
    if not documents:
        return []

    if len(documents) <= top_k:
        return documents

    query_tokens = set(tokenize(query))

    ranked = [
        (
            doc,
            _doc_relevance(query_tokens, doc),
        )
        for doc in documents
    ]

    selected: list[Document] = []
    candidates = list(ranked)

    while candidates and len(selected) < top_k:
        best_doc: Document | None = None
        best_score = float("-inf")
        best_idx = 0

        for idx, (doc, relevance) in enumerate(candidates):
            diversity_penalty = 0.0
            if selected:
                diversity_penalty = max(
                    _lexical_similarity(doc.page_content, chosen.page_content)
                    for chosen in selected
                )

            mmr_score = (lambda_mult * relevance) - ((1 - lambda_mult) * diversity_penalty)
            if mmr_score > best_score:
                best_score = mmr_score
                best_doc = doc
                best_idx = idx

        if best_doc is None:
            break

        selected.append(best_doc)
        candidates.pop(best_idx)

    return selected


def normalize_filters(filters: dict | None) -> dict | None:
    if not filters:
        return None

    payload = {}
    for key, value in filters.items():
        if value in (None, "", []):
            continue
        payload[key] = value
    return payload or None


def build_snippet(text: str, query: str, max_chars: int = 240) -> str:
    """Extract compact snippet around first matching query token."""
    cleaned_text = " ".join(text.split())
    if len(cleaned_text) <= max_chars:
        return cleaned_text

    query_tokens = [token for token in tokenize(query) if token not in STOPWORDS]
    lower_text = cleaned_text.lower()

    index = -1
    for token in query_tokens:
        index = lower_text.find(token.lower())
        if index >= 0:
            break

    if index < 0:
        return cleaned_text[: max_chars - 3].strip() + "..."

    half_window = max_chars // 2
    start = max(0, index - half_window)
    end = min(len(cleaned_text), start + max_chars)
    snippet = cleaned_text[start:end].strip()

    if start > 0:
        snippet = "..." + snippet
    if end < len(cleaned_text):
        snippet += "..."
    return snippet


def extract_filter_payload(filters: object | None) -> dict | None:
    """Convert pydantic model or dict to plain filter dict."""
    if filters is None:
        return None
    if hasattr(filters, "model_dump"):
        return normalize_filters(filters.model_dump())
    if isinstance(filters, dict):
        return normalize_filters(filters)
    return None


def source_summary(documents: Iterable[Document]) -> list[dict[str, object]]:
    """Small source digest for structured retrieval logging."""
    digest = []
    for doc in documents:
        digest.append(
            {
                "id": str(doc.metadata.get("id", "")),
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "score": doc.metadata.get("rerank_score")
                or doc.metadata.get("hybrid_score")
                or doc.metadata.get("distance"),
            }
        )
    return digest
