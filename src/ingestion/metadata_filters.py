"""Utilities to build SQL metadata filters for document retrieval."""

from typing import Any


def normalize_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    """Drop empty values before building SQL filter clauses."""
    if not filters:
        return {}

    normalized: dict[str, Any] = {}
    for key, value in filters.items():
        if value is None:
            continue
        if isinstance(value, list):
            cleaned = [item for item in value if item not in (None, "")]
            if cleaned:
                normalized[key] = cleaned
            continue
        if value == "":
            continue
        normalized[key] = value
    return normalized


def build_metadata_filter_clause(
    filters: dict[str, Any] | None,
    *,
    param_prefix: str = "",
) -> tuple[str, dict[str, Any]]:
    """Build SQL and params for metadata-based filtering."""
    clean_filters = normalize_filters(filters)
    if not clean_filters:
        return "", {}

    clauses: list[str] = []
    params: dict[str, Any] = {}

    def with_prefix(name: str) -> str:
        return f"{param_prefix}{name}" if param_prefix else name

    sources = clean_filters.get("sources")
    if isinstance(sources, list) and sources:
        source_placeholders: list[str] = []
        for idx, source in enumerate(sources):
            key = with_prefix(f"source_{idx}")
            params[key] = source
            source_placeholders.append(f"metadata->>'source' = :{key}")
        clauses.append("(" + " OR ".join(source_placeholders) + ")")

    source_type = clean_filters.get("source_type")
    if isinstance(source_type, str):
        key = with_prefix("source_type")
        params[key] = source_type
        clauses.append(f"metadata->>'source_type' = :{key}")

    page_from = clean_filters.get("page_from")
    if isinstance(page_from, int):
        key = with_prefix("page_from")
        params[key] = page_from
        clauses.append(f"(metadata->>'page')::int >= :{key}")

    page_to = clean_filters.get("page_to")
    if isinstance(page_to, int):
        key = with_prefix("page_to")
        params[key] = page_to
        clauses.append(f"(metadata->>'page')::int <= :{key}")

    chunk_types = clean_filters.get("chunk_types")
    if isinstance(chunk_types, list) and chunk_types:
        chunk_placeholders: list[str] = []
        for idx, chunk_type in enumerate(chunk_types):
            key = with_prefix(f"chunk_type_{idx}")
            params[key] = chunk_type
            chunk_placeholders.append(f"metadata->>'chunk_type' = :{key}")
        clauses.append("(" + " OR ".join(chunk_placeholders) + ")")

    content_origin = clean_filters.get("content_origin")
    if isinstance(content_origin, str):
        key = with_prefix("content_origin")
        params[key] = content_origin
        clauses.append(f"metadata->>'content_origin' = :{key}")

    if not clauses:
        return "", {}

    return " AND " + " AND ".join(clauses), params
