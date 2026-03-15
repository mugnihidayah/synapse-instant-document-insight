"""
Agent tools for Agentic RAG.

Each tool wraps existing infrastructure and returns structured results
that the agent orchestrator can use in its reasoning loop.
"""

import uuid

from langchain_core.documents import Document as LangchainDocument
from langchain_groq import ChatGroq
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.logger import get_logger
from src.ingestion.pgvector_store import similarity_search
from src.rag.grounding import compute_grounding_score
from src.rag.hybrid_search import hybrid_search
from src.rag.query_rewrite import rewrite_query
from src.rag.reranker import get_reranker
from src.rag.retrieval_utils import apply_mmr_diversification

logger = get_logger(__name__)


# RETRIEVE
async def tool_retrieve(
    db: AsyncSession,
    session_id: uuid.UUID,
    query: str,
    top_k: int = 5,
    filters: dict | None = None,
) -> list[LangchainDocument]:
    """
    Retrieve relevant document chunks using hybrid search + reranking.

    This is the core retrieval tool that wraps the existing pipeline:
    hybrid_search -> rerank -> MMR diversification.
    """
    fetch_k = max(top_k, settings.retrieval_fetch_k)

    if settings.use_hybrid_search:
        docs = await hybrid_search(
            db, session_id, query,
            k=fetch_k,
            vector_weight=settings.hybrid_vector_weight,
            keyword_weight=settings.hybrid_keyword_weight,
            filters=filters,
        )
    else:
        docs = await similarity_search(
            db, session_id, query,
            k=fetch_k,
            filters=filters,
        )

    reranker = get_reranker()
    docs = await reranker.rerank(query, docs, top_k=top_k)

    if settings.use_mmr:
        docs = apply_mmr_diversification(
            docs, query,
            top_k=top_k,
            lambda_mult=settings.mmr_lambda,
        )
    else:
        docs = docs[:top_k]

    logger.info("tool_retrieve", query=query[:80], results=len(docs))
    return docs


def format_retrieved_docs(docs: list[LangchainDocument]) -> str:
    """Format retrieved documents into a readable string for the agent."""
    if not docs:
        return "No documents found."

    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        preview = doc.page_content[:300]
        parts.append(f"[{i}] Source: {source}, Page: {page}\n{preview}")

    return "\n\n".join(parts)


# ANALYZE SOURCES
async def tool_analyze_sources(
    question: str,
    docs: list[LangchainDocument],
) -> dict:
    """
    Analyze if retrieved sources are sufficient to answer the question.

    Uses lexical grounding score + heuristic checks.
    """
    if not docs:
        return {
            "sufficient": False,
            "grounding_score": 0.0,
            "source_count": 0,
            "recommendation": "No documents retrieved. Try a different search query.",
        }

    source_texts = [doc.page_content for doc in docs]
    score = compute_grounding_score(question, source_texts)

    sufficient = score >= settings.groundedness_threshold and len(docs) >= 2

    if sufficient:
        recommendation = "Sources appear sufficient. Proceed to answer."
    elif score < settings.groundedness_threshold:
        recommendation = "Low relevance overlap. Consider refining the query."
    else:
        recommendation = "Limited sources. Consider searching with different terms."

    return {
        "sufficient": sufficient,
        "grounding_score": round(score, 4),
        "source_count": len(docs),
        "recommendation": recommendation,
    }


# SUMMARIZE CONTEXT
async def tool_summarize_context(
    context: str,
    focus: str = "",
) -> str:
    """
    Summarize retrieved context, optionally focusing on a specific aspect.

    Makes a lightweight LLM call to compress information.
    """
    llm = ChatGroq(
        api_key=SecretStr(settings.groq_api_key),
        model=settings.llm_model,
        temperature=0.1,
    )

    focus_instruction = f" Focus on: {focus}" if focus else ""

    prompt = (
        f"Summarize the following document context concisely.{focus_instruction}\n\n"
        f"Context:\n{context[:4000]}\n\n"
        f"Summary:"
    )

    response = await llm.ainvoke(prompt)
    content = response.content
    result = content.strip() if isinstance(content, str) else str(content)

    logger.info("tool_summarize", input_len=len(context), output_len=len(result))
    return result


# REFINE QUERY
async def tool_refine_query(
    original_query: str,
    context_so_far: str,
    reason: str = "",
) -> str:
    """
    Refine the search query based on what has been found so far.

    Combines rule-based rewriting with LLM reformulation.
    """
    # First apply rule-based rewriting
    cleaned = rewrite_query(original_query)

    # Then use LLM to reformulate based on context
    llm = ChatGroq(
        api_key=SecretStr(settings.groq_api_key),
        model=settings.llm_model,
        temperature=0.0,
    )

    reason_text = f" Reason for refinement: {reason}" if reason else ""

    prompt = (
        f"Given the original question and what has been found so far, "
        f"generate a better search query to find missing information.{reason_text}\n\n"
        f"Original question: {cleaned}\n\n"
        f"Information found so far:\n{context_so_far[:2000]}\n\n"
        f"Refined search query (just the query, nothing else):"
    )

    response = await llm.ainvoke(prompt)
    content = response.content
    refined = content.strip() if isinstance(content, str) else str(content)

    # Remove quotes if LLM wrapped the query
    refined = refined.strip("\"'")

    logger.info(
        "tool_refine_query",
        original=original_query[:50],
        refined=refined[:50],
    )
    return refined


# COMPARE SOURCES
async def tool_compare_sources(
    question: str,
    docs: list[LangchainDocument],
) -> str:
    """
    Cross-reference information across multiple chunks.

    Identifies agreements, contradictions, and gaps.
    """
    if len(docs) < 2:
        return "Not enough sources to compare (need at least 2)."

    llm = ChatGroq(
        api_key=SecretStr(settings.groq_api_key),
        model=settings.llm_model,
        temperature=0.1,
    )

    sources_text = ""
    for i, doc in enumerate(docs[:5], 1):
        source = doc.metadata.get("source", "unknown")
        sources_text += f"\n--- Source {i} ({source}) ---\n{doc.page_content[:800]}\n"

    prompt = (
        f"Compare the following sources regarding the question: {question}\n\n"
        f"Sources:{sources_text}\n\n"
        f"Analysis (mention agreements, contradictions, and information gaps):"
    )

    response = await llm.ainvoke(prompt)
    content = response.content
    result = content.strip() if isinstance(content, str) else str(content)

    logger.info("tool_compare", source_count=len(docs))
    return result


# TOOL REGISTRY
# Descriptions used by the agent to understand what each tool does
TOOL_DESCRIPTIONS: list[dict] = [
    {
        "name": "retrieve",
        "description": (
            "Search the uploaded documents with a query. "
            "Returns the most relevant document chunks ranked by relevance. "
            "Use this when you need to find information from the documents."
        ),
        "parameters": {
            "query": "The search query string",
            "top_k": "Number of results to return (default: 5)",
        },
    },
    {
        "name": "analyze_sources",
        "description": (
            "Evaluate whether the currently retrieved sources are sufficient "
            "to answer the user's question. Returns a sufficiency assessment "
            "with a recommendation."
        ),
        "parameters": {},
    },
    {
        "name": "summarize_context",
        "description": (
            "Summarize the retrieved context into a concise overview. "
            "Use this when you have too much information and need to condense it."
        ),
        "parameters": {
            "focus": "Optional aspect to focus the summary on",
        },
    },
    {
        "name": "refine_query",
        "description": (
            "Generate a better search query based on what has been found so far. "
            "Use this when initial retrieval didn't return relevant results."
        ),
        "parameters": {
            "reason": "Why the current results are insufficient",
        },
    },
    {
        "name": "compare_sources",
        "description": (
            "Cross-reference information across retrieved document chunks. "
            "Identifies agreements, contradictions, and gaps in the sources. "
            "Use this when you need to verify consistency across sources."
        ),
        "parameters": {},
    },
]
