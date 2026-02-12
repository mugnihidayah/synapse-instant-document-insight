"""
Tests for RAG module
"""

import pytest

from src.rag.prompts import PROMPT_EN, PROMPT_ID, get_prompt


class TestPrompts:
    """Tests for prompt templates"""

    def test_prompt_en_exists(self) -> None:
        """Test that PROMPT_EN exists and is not empty"""
        assert PROMPT_EN is not None
        assert isinstance(PROMPT_EN, str)
        assert len(PROMPT_EN) > 100

    def test_prompt_id_exists(self) -> None:
        """Test that PROMPT_ID exists and is not empty"""
        assert PROMPT_ID is not None
        assert isinstance(PROMPT_ID, str)
        assert len(PROMPT_ID) > 100

    def test_prompt_en_has_placeholders(self) -> None:
        """Test that English prompt has required placeholders."""
        assert "{context}" in PROMPT_EN
        assert "{question}" in PROMPT_EN
        assert "{chat_history}" in PROMPT_EN

    def test_prompt_id_has_placeholders(self) -> None:
        """Test that Indonesian prompt has required placeholders."""
        assert "{context}" in PROMPT_ID
        assert "{question}" in PROMPT_ID
        assert "{chat_history}" in PROMPT_ID

    def test_get_prompt_english(self) -> None:
        """Test getting English prompt."""
        prompt = get_prompt("en")
        assert prompt == PROMPT_EN

    def test_get_prompt_indonesian(self) -> None:
        """Test getting Indonesian prompt."""
        prompt = get_prompt("id")
        assert prompt == PROMPT_ID

    def test_get_prompt_default_is_indonesian(self) -> None:
        """Test that default language is Indonesian."""
        prompt = get_prompt()
        assert prompt == PROMPT_ID

    def test_get_prompt_unknown_language_returns_indonesian(self) -> None:
        """Test that unknown language falls back to Indonesian."""
        prompt = get_prompt("fr")  # French not supported
        assert prompt == PROMPT_ID


class TestReranker:
    """Tests for reranker module."""

    def test_get_reranker_returns_ranker(self) -> None:
        """Test that get_reranker returns a Ranker instance."""
        from src.rag.reranker import get_reranker

        # This may load the model, so it's slow
        reranker = get_reranker()

        assert reranker is not None
        assert hasattr(reranker, "rerank")

    def test_get_reranker_cached(self) -> None:
        """Test that reranker is cached (same instance returned)."""
        from src.rag.reranker import get_reranker

        r1 = get_reranker()
        r2 = get_reranker()

        # Should be same instance due to caching
        assert r1 is r2

    @pytest.mark.asyncio
    async def test_rerank_empty_list(self) -> None:
        """Test reranking empty list returns empty list."""
        from src.rag.reranker import get_reranker

        reranker = get_reranker()
        result = await reranker.rerank("query", [])
        assert result == []

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_rerank_returns_documents(self) -> None:
        """Test that reranking returns documents."""
        from langchain_core.documents import Document

        from src.rag.reranker import get_reranker

        docs = [
            Document(page_content="Python is a programming language.", metadata={"id": "1"}),
            Document(
                page_content="Java is also a programming language.",
                metadata={"id": "2"},
            ),
            Document(page_content="The weather is nice today.", metadata={"id": "3"}),
        ]

        reranker = get_reranker()
        result = await reranker.rerank("Python programming", docs, top_k=2)

        assert len(result) <= 2
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_rerank_respects_top_k(self) -> None:
        """Test that top_k limits results."""
        from langchain_core.documents import Document

        from src.rag.reranker import get_reranker

        docs = [
            Document(page_content="Doc 1", metadata={"id": "1"}),
            Document(page_content="Doc 2", metadata={"id": "2"}),
            Document(page_content="Doc 3", metadata={"id": "3"}),
        ]

        reranker = get_reranker()
        result = await reranker.rerank("query", docs, top_k=1)

        assert len(result) <= 1


class TestChainIntegration:
    """Integration tests for RAG chain (requires API keys)."""

    @pytest.mark.skip(reason="Requires actual API keys")
    def test_full_rag_chain(self) -> None:
        """
        Test full RAG chain with real vectorstore and LLM.

        This test is skipped by default as it requires:
        - Valid GROQ_API_KEY
        - Valid HUGGINGFACE_TOKEN
        - Network access
        """
        pass
