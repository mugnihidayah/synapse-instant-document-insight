"""Lightweight query rewriting for better retrieval recall."""

import re

from src.core.config import settings

FILLER_PREFIXES = [
    r"^tolong\s+",
    r"^please\s+",
    r"^bisa\s+",
    r"^can\s+you\s+",
    r"^could\s+you\s+",
    r"^mohon\s+",
]


def rewrite_query(question: str, enabled: bool = True) -> str:
    """Normalize and slightly denoise a question before retrieval."""
    if not enabled or not settings.query_rewrite_enabled:
        return question.strip()

    rewritten = " ".join(question.strip().split())
    lowered = rewritten.lower()

    for pattern in FILLER_PREFIXES:
        lowered = re.sub(pattern, "", lowered)

    # Keep punctuation that helps semantics; remove trailing filler punctuation.
    lowered = lowered.strip(" ?.!")

    if not lowered:
        return question.strip()

    return lowered
