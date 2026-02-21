"""Grounding checks for generated answers against retrieved sources."""

from src.core.config import settings
from src.rag.retrieval_utils import STOPWORDS, tokenize


def compute_grounding_score(answer: str, source_texts: list[str]) -> float:
    """
    Lexical grounding proxy: overlap between answer tokens and source tokens.

    This is intentionally lightweight so it can run on free-tier resources.
    """
    answer_tokens = [token for token in tokenize(answer) if token not in STOPWORDS]
    if not answer_tokens:
        return 0.0

    source_vocab: set[str] = set()
    for text in source_texts:
        source_vocab.update(tokenize(text))

    if not source_vocab:
        return 0.0

    matched = sum(1 for token in answer_tokens if token in source_vocab)
    return round(matched / len(answer_tokens), 4)


def is_grounded(
    answer: str, source_texts: list[str], threshold: float | None = None
) -> tuple[bool, float]:
    score = compute_grounding_score(answer, source_texts)
    limit = settings.groundedness_threshold if threshold is None else threshold
    return score >= limit, score


def build_low_grounding_fallback(language: str) -> str:
    if language == "en":
        return (
            "I cannot confidently answer from the uploaded documents. "
            "Please refine your question or adjust the source filters."
        )
    return (
        "Saya belum bisa menjawab dengan yakin dari dokumen yang diunggah. "
        "Silakan perjelas pertanyaan atau ubah filter sumber."
    )
