"""RAG query endpoints with streaming."""

import json
import math
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.documents import Document as LangchainDocument
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import Request

from src.api import session as session_service
from src.api.dependencies import get_api_key
from src.api.rate_limiter import RATE_LIMIT_QUERY, limiter
from src.api.schemas import QueryDebug, QueryRequest, QueryResponse, SourceItem
from src.api.usage import has_remaining_query_quota, record_usage_event
from src.core.config import settings
from src.core.logger import get_logger
from src.db import get_db
from src.db.models import APIKey
from src.ingestion.pgvector_store import similarity_search
from src.rag.chain import create_rag_chain
from src.rag.chat_history import format_chat_history, get_chat_history, save_chat_message
from src.rag.contextualize import contextualize_query
from src.rag.grounding import build_low_grounding_fallback, is_grounded
from src.rag.hybrid_search import hybrid_search
from src.rag.query_rewrite import rewrite_query
from src.rag.reranker import get_reranker
from src.rag.retrieval_utils import (
    apply_mmr_diversification,
    build_snippet,
    compute_dynamic_top_k,
    extract_filter_payload,
    source_summary,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


def _score_document(meta: dict) -> float:
    rerank_score = meta.get("rerank_score")
    if isinstance(rerank_score, (int, float)):
        # squash cross-encoder score to [0, 1]
        return max(0.0, min(1.0, 1 / (1 + math.exp(-float(rerank_score)))))

    hybrid_score = meta.get("hybrid_score")
    if isinstance(hybrid_score, (int, float)):
        return max(0.0, min(1.0, float(hybrid_score) / 0.03))

    distance = meta.get("distance")
    if isinstance(distance, (int, float)):
        return max(0.0, min(1.0, 1.0 - float(distance)))

    return 0.0


def _build_sources(docs: list[LangchainDocument], query: str) -> list[SourceItem]:
    """Build enriched source items from retrieved documents."""
    sources = []
    for doc in docs:
        meta = dict(doc.metadata)

        chunk_id = str(meta.pop("id", ""))
        meta.pop("distance", None)
        meta.pop("hybrid_score", None)
        meta.pop("keyword_rank", None)
        meta.pop("rerank_score", None)

        sources.append(
            SourceItem(
                text=doc.page_content,
                snippet=build_snippet(doc.page_content, query),
                score=round(_score_document(doc.metadata), 4),
                chunk_id=chunk_id,
                source=meta.get("source"),
                page=meta.get("page"),
                metadata=meta,
            )
        )
    return sources


async def _retrieve_docs(
    db: AsyncSession,
    session_id: uuid.UUID,
    query_request: QueryRequest,
    chat_history: str,
) -> tuple[list[LangchainDocument], QueryDebug, str]:
    contextualized_question = await contextualize_query(
        query_request.question,
        chat_history,
        query_request.model,
    )
    rewritten_query = rewrite_query(
        contextualized_question,
        enabled=query_request.enable_query_rewrite,
    )

    filters = extract_filter_payload(query_request.filters)

    top_k = compute_dynamic_top_k(rewritten_query, query_request.top_k)
    top_k = max(1, min(top_k, 50))

    fetch_k = max(top_k, settings.retrieval_fetch_k)

    if settings.use_hybrid_search:
        docs = await hybrid_search(
            db,
            session_id,
            rewritten_query,
            k=fetch_k,
            vector_weight=settings.hybrid_vector_weight,
            keyword_weight=settings.hybrid_keyword_weight,
            filters=filters,
        )
    else:
        docs = await similarity_search(
            db,
            session_id,
            rewritten_query,
            k=fetch_k,
            filters=filters,
        )

    rerank_top_k = query_request.rerank_top_k or settings.rerank_top_k
    rerank_top_k = max(1, min(rerank_top_k, top_k))

    reranker = get_reranker()
    docs = await reranker.rerank(
        rewritten_query,
        docs,
        top_k=max(rerank_top_k, top_k),
    )

    if settings.use_mmr:
        docs = apply_mmr_diversification(
            docs,
            rewritten_query,
            top_k=rerank_top_k,
            lambda_mult=settings.mmr_lambda,
        )
    else:
        docs = docs[:rerank_top_k]

    logger.info(
        "retrieval_completed",
        session_id=str(session_id),
        query=rewritten_query[:120],
        filters=filters,
        top_k=top_k,
        rerank_top_k=rerank_top_k,
        source_digest=source_summary(docs)[:5],
    )

    return (
        docs,
        QueryDebug(
            rewritten_query=rewritten_query,
            retrieved_count=fetch_k,
            reranked_count=len(docs),
            top_k_used=top_k,
            rerank_top_k_used=rerank_top_k,
            filters_applied=filters,
        ),
        rewritten_query,
    )


@router.post("/stream/{session_id}")
@limiter.limit(RATE_LIMIT_QUERY)
async def query_stream(
    request: Request,
    session_id: str,
    query_request: QueryRequest,
    db: AsyncSession = Depends(get_db),
    api_key: APIKey = Depends(get_api_key),
) -> StreamingResponse:
    """Query with SSE streaming response."""
    session = await session_service.get_session_by_str(db, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    if session.api_key_id != api_key.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this session",
        )

    if session.document_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents in session. Upload documents first.",
        )

    if not await has_remaining_query_quota(db, api_key.id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Daily query quota reached. Please try again tomorrow.",
        )

    chat_messages = await get_chat_history(db, session.id, limit=5)
    chat_history_str = format_chat_history(chat_messages)

    try:
        docs, debug_payload, rewritten_query = await _retrieve_docs(
            db,
            session.id,
            query_request,
            chat_history_str,
        )

        context = "\n\n".join([doc.page_content for doc in docs])
        sources = _build_sources(docs, rewritten_query)
        source_texts = [doc.page_content for doc in docs]

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {exc}",
        ) from exc

    async def generate() -> AsyncGenerator[str, None]:
        try:
            chain = create_rag_chain(
                model_name=query_request.model or settings.llm_model,
                temperature=query_request.temperature,
                language=query_request.language,
            )

            full_response = ""

            if query_request.strict_grounding:
                generated = await chain.ainvoke(
                    {
                        "context": context,
                        "question": query_request.question,
                        "chat_history": chat_history_str,
                    }
                )
                grounded, grounding_score = is_grounded(generated, source_texts)
                if not grounded:
                    generated = build_low_grounding_fallback(query_request.language)
                full_response = generated
                yield f"data: {json.dumps({'chunk': generated})}\n\n"
            else:
                async for chunk in chain.astream(
                    {
                        "context": context,
                        "question": query_request.question,
                        "chat_history": chat_history_str,
                    }
                ):
                    full_response += chunk
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                grounded, grounding_score = is_grounded(full_response, source_texts)

            await save_chat_message(db, session.id, "user", query_request.question)
            await save_chat_message(db, session.id, "assistant", full_response)

            await record_usage_event(
                db,
                api_key.id,
                "query",
                session_id=session.id,
                metadata={
                    "model": query_request.model or settings.llm_model,
                    "grounded": grounded,
                    "grounding_score": grounding_score,
                    "rewritten_query": rewritten_query,
                    "source_count": len(sources),
                    "top_k": debug_payload.top_k_used,
                    "rerank_top_k": debug_payload.rerank_top_k_used,
                },
            )

            payload = {
                "sources": [s.model_dump() for s in sources],
                "grounded": grounded,
                "grounding_score": grounding_score,
                "rewritten_query": rewritten_query,
            }
            if query_request.include_debug:
                payload["debug"] = debug_payload.model_dump()

            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/{session_id}", response_model=QueryResponse)
@limiter.limit(RATE_LIMIT_QUERY)
async def query(
    request: Request,
    session_id: str,
    query_request: QueryRequest,
    db: AsyncSession = Depends(get_db),
    api_key: APIKey = Depends(get_api_key),
) -> QueryResponse:
    """Query without streaming."""
    logger.info(
        "query_started",
        session_id=session_id,
        question=query_request.question[:80],
    )

    session = await session_service.get_session_by_str(db, session_id)
    if not session:
        logger.warning("query_failed", reason="session_not_found", session_id=session_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    if session.api_key_id != api_key.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this session",
        )

    if session.document_count == 0:
        logger.warning("query_failed", reason="no_documents", session_id=session_id)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents in session",
        )

    if not await has_remaining_query_quota(db, api_key.id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Daily query quota reached. Please try again tomorrow.",
        )

    chat_messages = await get_chat_history(db, session.id, limit=5)
    chat_history_str = format_chat_history(chat_messages)

    try:
        docs, debug_payload, rewritten_query = await _retrieve_docs(
            db,
            session.id,
            query_request,
            chat_history_str,
        )

        context = "\n\n".join([doc.page_content for doc in docs])

        chain = create_rag_chain(
            model_name=query_request.model or settings.llm_model,
            temperature=query_request.temperature,
            language=query_request.language,
        )

        response = await chain.ainvoke(
            {
                "context": context,
                "question": query_request.question,
                "chat_history": chat_history_str,
            }
        )

        source_texts = [doc.page_content for doc in docs]
        grounded, grounding_score = is_grounded(response, source_texts)
        if query_request.strict_grounding and not grounded:
            response = build_low_grounding_fallback(query_request.language)

        sources = _build_sources(docs, rewritten_query)

        logger.info(
            "query_completed",
            session_id=session_id,
            sources_count=len(sources),
            grounded=grounded,
            grounding_score=grounding_score,
        )

        await save_chat_message(db, session.id, "user", query_request.question)
        await save_chat_message(db, session.id, "assistant", response)

        await record_usage_event(
            db,
            api_key.id,
            "query",
            session_id=session.id,
            metadata={
                "model": query_request.model or settings.llm_model,
                "grounded": grounded,
                "grounding_score": grounding_score,
                "rewritten_query": rewritten_query,
                "source_count": len(sources),
                "top_k": debug_payload.top_k_used,
                "rerank_top_k": debug_payload.rerank_top_k_used,
            },
        )

        return QueryResponse(
            answer=response,
            sources=sources,
            model_used=query_request.model or settings.llm_model,
            rewritten_query=rewritten_query,
            grounded=grounded,
            grounding_score=grounding_score,
            debug=debug_payload if query_request.include_debug else None,
        )

    except Exception as exc:
        logger.error(
            "query_failed",
            session_id=session_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
