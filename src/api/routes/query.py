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

from src.agent.orchestrator import run_agent
from src.api import session as session_service
from src.api.dependencies import get_api_key
from src.api.rate_limiter import RATE_LIMIT_QUERY, limiter
from src.api.schemas import AgentStepResponse, QueryDebug, QueryRequest, QueryResponse, SourceItem
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


def _clamp_01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _coerce_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []

    minimum = min(values)
    maximum = max(values)
    spread = maximum - minimum

    if spread < 1e-9:
        return [0.5 for _ in values]

    return [_clamp_01((item - minimum) / spread) for item in values]


def _collect_indexed_scores(values: list[float | None], indices: list[int]) -> list[float]:
    """Collect non-null scores from selected positions in a stable order."""
    collected: list[float] = []
    for idx in indices:
        value = values[idx]
        if value is not None:
            collected.append(value)
    return collected


def _sigmoid(value: float) -> float:
    # Guard against overflow in extreme local reranker outputs.
    if value >= 40:
        return 1.0
    if value <= -40:
        return 0.0
    return 1 / (1 + math.exp(-value))


def _compute_display_scores(docs: list[LangchainDocument]) -> list[float]:
    """
    Compute citation scores using both reranker and retrieval signals.

    Why:
    - Avoid flat 0.50 scores when reranker outputs identical/near-identical values.
    - Keep ranking signal meaningful across different reranker providers.
    """
    if not docs:
        return []

    rerank_raw: list[float | None] = []
    retrieval_raw: list[float | None] = []

    for doc in docs:
        meta = doc.metadata
        rerank_raw.append(_coerce_float(meta.get("rerank_score")))

        hybrid_score = _coerce_float(meta.get("hybrid_score"))
        if hybrid_score is not None:
            retrieval_raw.append(hybrid_score)
            continue

        distance = _coerce_float(meta.get("distance"))
        if distance is not None:
            retrieval_raw.append(1.0 - distance)
            continue

        keyword_rank = _coerce_float(meta.get("keyword_rank"))
        retrieval_raw.append(keyword_rank)

    retrieval_scores = [0.0 for _ in docs]
    retrieval_idx = [idx for idx, score in enumerate(retrieval_raw) if score is not None]
    if retrieval_idx:
        retrieval_norm = _normalize(_collect_indexed_scores(retrieval_raw, retrieval_idx))
        for idx, score in zip(retrieval_idx, retrieval_norm, strict=True):
            retrieval_scores[idx] = score
    elif len(docs) == 1:
        retrieval_scores[0] = 1.0
    else:
        denominator = max(1, len(docs) - 1)
        for idx in range(len(docs)):
            retrieval_scores[idx] = 1.0 - (idx / denominator)

    rerank_idx = [idx for idx, score in enumerate(rerank_raw) if score is not None]
    if not rerank_idx:
        return [round(_clamp_01(score), 4) for score in retrieval_scores]

    rerank_values = _collect_indexed_scores(rerank_raw, rerank_idx)
    if all(0.0 <= value <= 1.0 for value in rerank_values):
        rerank_base = rerank_values
    else:
        rerank_base = [_sigmoid(value) for value in rerank_values]
    rerank_norm = _normalize(rerank_base)

    scores = []
    rerank_position = {doc_idx: pos for pos, doc_idx in enumerate(rerank_idx)}
    for idx, retrieval_score in enumerate(retrieval_scores):
        position = rerank_position.get(idx)
        if position is None:
            score = retrieval_score
        else:
            # Weighted blend makes scores less flat while keeping rerank dominant.
            score = (0.75 * rerank_norm[position]) + (0.25 * retrieval_score)
        scores.append(round(_clamp_01(score), 4))

    return scores


def _build_sources(docs: list[LangchainDocument], query: str) -> list[SourceItem]:
    """Build enriched source items from retrieved documents."""
    display_scores = _compute_display_scores(docs)
    sources = []
    for idx, doc in enumerate(docs):
        meta = dict(doc.metadata)
        document_id = meta.get("document_id")
        raw_rerank = _coerce_float(meta.get("rerank_score"))
        raw_hybrid = _coerce_float(meta.get("hybrid_score"))
        raw_distance = _coerce_float(meta.get("distance"))

        chunk_id = str(meta.pop("id", ""))
        meta.pop("distance", None)
        meta.pop("hybrid_score", None)
        meta.pop("keyword_rank", None)
        meta.pop("rerank_score", None)
        meta.pop("file_path", None)
        if raw_rerank is not None:
            meta["raw_rerank_score"] = round(raw_rerank, 6)
        if raw_hybrid is not None:
            meta["raw_hybrid_score"] = round(raw_hybrid, 6)
        if raw_distance is not None:
            meta["raw_distance"] = round(raw_distance, 6)

        sources.append(
            SourceItem(
                text=doc.page_content,
                snippet=build_snippet(doc.page_content, query),
                score=display_scores[idx],
                chunk_id=chunk_id,
                document_id=str(document_id) if document_id else None,
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


def _agent_to_response(result, query_request):
    """Convert AgentResult to QueryResponse."""
    agent_steps = [
        AgentStepResponse(
            step_type=step.step_type,
            content=step.content,
            tool_name=step.tool_name,
        )
        for step in result.steps
    ]

    return QueryResponse(
        answer=result.answer,
        sources=[SourceItem(**s) for s in result.sources],
        model_used=result.model_used,
        rewritten_query=result.rewritten_query,
        grounded=result.grounded,
        grounding_score=result.grounding_score,
        debug=None,
        agent_steps=agent_steps,
        agent_iterations=result.iterations,
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

    if query_request.agent_mode:
        async def generate_agent() -> AsyncGenerator[str, None]:
            try:
                result = await run_agent(
                    question=query_request.question,
                    session_id=session.id,
                    db=db,
                    language=query_request.language,
                    model_name=query_request.model or settings.llm_model,
                    max_iterations=query_request.max_agent_steps,
                    temperature=query_request.temperature,
                    filters=extract_filter_payload(query_request.filters),
                    chat_history_str=chat_history_str,
                )

                # Stream agent steps
                for step in result.steps:
                    step_data = {
                        "step": {
                            "step_type": step.step_type,
                            "content": step.content,
                            "tool_name": step.tool_name,
                        }
                    }
                    yield f"data: {json.dumps(step_data)}\n\n"

                # Stream final answer as chunks
                yield f"data: {json.dumps({'chunk': result.answer})}\n\n"

                await save_chat_message(db, session.id, "user", query_request.question)
                await save_chat_message(db, session.id, "assistant", result.answer)

                await record_usage_event(
                    db,
                    api_key.id,
                    "query",
                    session_id=session.id,
                    metadata={
                        "model": result.model_used,
                        "agent_mode": True,
                        "agent_iterations": result.iterations,
                        "grounded": result.grounded,
                        "grounding_score": result.grounding_score,
                        "source_count": len(result.sources),
                    },
                )

                # Send sources + agent metadata
                agent_steps_data = [
                    {
                        "step_type": s.step_type,
                        "content": s.content,
                        "tool_name": s.tool_name,
                    }
                    for s in result.steps
                ]

                payload = {
                    "sources": [s.model_dump() for s in result.sources],
                    "grounded": result.grounded,
                    "grounding_score": result.grounding_score,
                    "agent_steps": agent_steps_data,
                    "agent_iterations": result.iterations,
                }
                yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(generate_agent(), media_type="text/event-stream")

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

    if query_request.agent_mode:
        try:
            filters = extract_filter_payload(query_request.filters)
            result = await run_agent(
                question=query_request.question,
                session_id=session.id,
                db=db,
                language=query_request.language,
                model_name=query_request.model or settings.llm_model,
                max_iterations=query_request.max_agent_steps,
                temperature=query_request.temperature,
                filters=filters,
                chat_history_str=chat_history_str,
            )

            await save_chat_message(db, session.id, "user", query_request.question)
            await save_chat_message(db, session.id, "assistant", result.answer)

            await record_usage_event(
                db,
                api_key.id,
                "query",
                session_id=session.id,
                metadata={
                    "model": result.model_used,
                    "agent_mode": True,
                    "agent_iterations": result.iterations,
                    "grounded": result.grounded,
                    "grounding_score": result.grounding_score,
                    "source_count": len(result.sources),
                },
            )

            return _agent_to_response(result, query_request)

        except Exception as exc:
            logger.error("agent_query_failed", error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

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
