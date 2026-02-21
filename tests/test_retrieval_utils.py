"""Tests for retrieval utilities."""

from langchain_core.documents import Document

from src.rag.retrieval_utils import (
    apply_mmr_diversification,
    build_snippet,
    compute_dynamic_top_k,
    extract_filter_payload,
)


def test_compute_dynamic_top_k_uses_requested_value() -> None:
    assert compute_dynamic_top_k("short query", requested_top_k=7) == 7


def test_compute_dynamic_top_k_grows_with_complexity() -> None:
    simple = compute_dynamic_top_k("apa itu rag")
    complex_q = compute_dynamic_top_k(
        "jelaskan metodologi evaluasi retrieval dan dampaknya ke kualitas grounded answer"
    )
    assert complex_q >= simple


def test_apply_mmr_diversification_returns_top_k() -> None:
    docs = [
        Document(page_content="Python improves developer productivity", metadata={}),
        Document(page_content="Python has many libraries for data science", metadata={}),
        Document(page_content="Supply chain risk includes logistics delays", metadata={}),
    ]

    selected = apply_mmr_diversification(docs, "python libraries", top_k=2, lambda_mult=0.7)

    assert len(selected) == 2
    assert isinstance(selected[0], Document)


def test_build_snippet_shortens_long_text() -> None:
    text = "A" * 120 + " retrieval " + "B" * 120
    snippet = build_snippet(text, "retrieval", max_chars=80)
    assert len(snippet) <= 86
    assert "retrieval" in snippet.lower()


def test_extract_filter_payload_drops_empty_values() -> None:
    payload = extract_filter_payload({"sources": ["a.pdf"], "source_type": "", "page_from": 1})
    assert payload == {"sources": ["a.pdf"], "page_from": 1}
