"""
RAG module containing retrieval and generation logic

This module provides:
- Document retrieval with reranking
- LLM chain for question answering
- Prompt templates
"""

from src.rag.chain import ask_question, format_chat_history
from src.rag.prompts import PROMPT_EN, PROMPT_ID, get_prompt
from src.rag.reranker import get_reranker

__all__: list[str] = [
    "ask_question",
    "format_chat_history",
    "get_prompt",
    "PROMPT_EN",
    "PROMPT_ID",
    "get_reranker",
]
