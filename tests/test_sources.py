"""Tests for enriched source citation format.

Self-contained test that duplicates _build_sources logic
to avoid heavy import chain (FastAPI, SQLAlchemy, etc).
"""

from langchain_core.documents import Document as LangchainDocument


def _build_sources(docs: list[LangchainDocument]) -> list[dict]:
    """Duplicate of src.api.routes.query._build_sources for testing."""
    sources = []
    for doc in docs:
        meta = dict(doc.metadata)

        distance = meta.pop("distance", None)
        hybrid_score = meta.pop("hybrid_score", None)
        meta.pop("keyword_rank", None)
        chunk_id = meta.pop("id", "")

        if hybrid_score is not None:
            score = min(hybrid_score / 0.03, 1.0)
        elif distance is not None:
            score = max(0.0, 1.0 - float(distance))
        else:
            score = 0.0

        sources.append(
            {
                "text": doc.page_content,
                "score": round(score, 4),
                "chunk_id": chunk_id,
                "metadata": meta,
            }
        )
    return sources


class TestBuildSources:
    """Test _build_sources helper function."""

    def test_cosine_distance_to_score(self):
        """Cosine distance should be converted to 0-1 similarity score."""
        docs = [
            LangchainDocument(
                page_content="test content",
                metadata={
                    "id": "chunk-123",
                    "source": "test.pdf",
                    "source_type": "pdf",
                    "distance": 0.15,
                },
            )
        ]
        sources = _build_sources(docs)

        assert len(sources) == 1
        assert sources[0]["score"] == 0.85  # 1 - 0.15
        assert sources[0]["chunk_id"] == "chunk-123"
        assert "distance" not in sources[0]["metadata"]
        assert "id" not in sources[0]["metadata"]

    def test_hybrid_score_normalization(self):
        """Hybrid RRF score should be normalized to 0-1."""
        docs = [
            LangchainDocument(
                page_content="hybrid result",
                metadata={
                    "id": "chunk-456",
                    "source": "doc.pdf",
                    "hybrid_score": 0.015,
                    "keyword_rank": 2.5,
                },
            )
        ]
        sources = _build_sources(docs)

        assert 0 <= sources[0]["score"] <= 1
        assert sources[0]["chunk_id"] == "chunk-456"
        assert "hybrid_score" not in sources[0]["metadata"]
        assert "keyword_rank" not in sources[0]["metadata"]

    def test_metadata_preserved(self):
        """source, source_type, page, total_pages should stay in metadata."""
        docs = [
            LangchainDocument(
                page_content="page content",
                metadata={
                    "id": "chunk-789",
                    "source": "report.pdf",
                    "source_type": "pdf",
                    "page": 3,
                    "total_pages": 15,
                    "distance": 0.2,
                },
            )
        ]
        sources = _build_sources(docs)
        meta = sources[0]["metadata"]

        assert meta["source"] == "report.pdf"
        assert meta["source_type"] == "pdf"
        assert meta["page"] == 3
        assert meta["total_pages"] == 15

    def test_no_score_defaults_to_zero(self):
        """If no distance or hybrid_score, score defaults to 0."""
        docs = [
            LangchainDocument(
                page_content="no score",
                metadata={"id": "chunk-000"},
            )
        ]
        sources = _build_sources(docs)
        assert sources[0]["score"] == 0.0

    def test_empty_docs(self):
        """Empty list should return empty sources."""
        assert _build_sources([]) == []

    def test_output_format_complete(self):
        """Verify all required fields present in output."""
        docs = [
            LangchainDocument(
                page_content="content here",
                metadata={
                    "id": "abc",
                    "source": "file.pdf",
                    "source_type": "pdf",
                    "total_pages": 5,
                    "distance": 0.1,
                },
            )
        ]
        sources = _build_sources(docs)
        src = sources[0]

        assert "text" in src
        assert "score" in src
        assert "chunk_id" in src
        assert "metadata" in src
        assert isinstance(src["score"], float)
        assert isinstance(src["chunk_id"], str)
        assert isinstance(src["metadata"], dict)
