"""Tests for grounding checks."""

from src.rag.grounding import build_low_grounding_fallback, compute_grounding_score, is_grounded


def test_compute_grounding_score_high_for_overlapping_text() -> None:
    answer = "Pendapatan 2025 adalah 120 miliar rupiah"
    sources = ["Dokumen menyebut pendapatan 2025 sebesar 120 miliar rupiah."]
    score = compute_grounding_score(answer, sources)
    assert score > 0.5


def test_compute_grounding_score_low_for_unrelated_text() -> None:
    answer = "Harga bitcoin naik tajam"
    sources = ["Dokumen membahas kebijakan sumber daya manusia perusahaan."]
    score = compute_grounding_score(answer, sources)
    assert score < 0.4


def test_is_grounded_threshold_override() -> None:
    grounded, score = is_grounded(
        "growth is 10 percent",
        ["The report states growth is 10 percent year over year."],
        threshold=0.3,
    )
    assert grounded
    assert score >= 0.3


def test_fallback_messages_language() -> None:
    assert "cannot confidently answer" in build_low_grounding_fallback("en")
    assert "belum bisa menjawab" in build_low_grounding_fallback("id")
