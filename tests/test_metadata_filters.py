"""Tests for metadata filter SQL builder."""

from src.ingestion.metadata_filters import build_metadata_filter_clause


def test_build_metadata_filter_clause_empty() -> None:
    sql, params = build_metadata_filter_clause(None)
    assert sql == ""
    assert params == {}


def test_build_metadata_filter_clause_with_sources_and_pages() -> None:
    sql, params = build_metadata_filter_clause(
        {
            "sources": ["a.pdf", "b.pdf"],
            "page_from": 2,
            "page_to": 5,
        }
    )

    assert "metadata->>'source'" in sql
    assert "page" in sql
    assert params["source_0"] == "a.pdf"
    assert params["source_1"] == "b.pdf"
    assert params["page_from"] == 2
    assert params["page_to"] == 5


def test_build_metadata_filter_clause_skips_empty_values() -> None:
    sql, params = build_metadata_filter_clause({"sources": [], "source_type": ""})
    assert sql == ""
    assert params == {}
