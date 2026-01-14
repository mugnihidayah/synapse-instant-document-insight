"""
RAG module containing retrieval and generation logic

This module provides:
- Document retrieval with reranking
- LLM chain for question answering
- Prompt templates
"""

from src.rag.chain import ask_question, format_chat_history
from src.rag.prompts import get_prompt, PROMPT_EN, PROMPT_ID
from src.rag.reranker import rerank_documents, get_reranker

__all__: list[str] = [
  "ask_question",
  "format_chat_history",
  "get_prompt",
  "PROMPT_EN",
  "PROMPT_ID",
  "rerank_documents",
  "get_reranker",
]