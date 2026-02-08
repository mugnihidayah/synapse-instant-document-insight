"""
RAG query endpoints with streaming
"""

import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import Request

from src.api import session as session_service
from src.api.dependencies import get_api_key
from src.api.rate_limiter import RATE_LIMIT_QUERY, limiter
from src.api.schemas import QueryRequest, QueryResponse
from src.core.config import settings
from src.core.logger import get_logger
from src.db import get_db
from src.db.models import APIKey
from src.ingestion.pgvector_store import similarity_search
from src.rag.chain import create_rag_chain
from src.rag.chat_history import (
    format_chat_history,
    get_chat_history,
    save_chat_message,
)
from src.rag.contextualize import contextualize_query
from src.rag.reranker import get_reranker

logger = get_logger(__name__)


router = APIRouter(prefix="/query", tags=["Query"])


@router.post("/stream/{session_id}")
@limiter.limit(RATE_LIMIT_QUERY)
async def query_stream(
    request: Request,
    session_id: str,
    query_request: QueryRequest,
    db: AsyncSession = Depends(get_db),
    api_key: APIKey = Depends(get_api_key),
) -> StreamingResponse:
    """Query with streaming response"""

    # Validate session
    session = await session_service.get_session_by_str(db, session_id)

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    if session.document_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents in session. Upload documents first.",
        )

    # Get chat history
    chat_messages = await get_chat_history(db, session.id, limit=5)
    chat_history_str = format_chat_history(chat_messages)

    # Contextualize query
    contextualized_question = await contextualize_query(
        query_request.question,
        chat_history_str,
        query_request.model,
    )

    try:
        # Get relevant documents
        docs = await similarity_search(
            db,
            session.id,
            contextualized_question,
            k=settings.retrieval_top_k,
        )

        # Rerank documents
        reranker = get_reranker()
        docs = await reranker.rerank(
            contextualized_question,
            docs,
            top_k=settings.rerank_top_k,
        )

        # Format context
        context = "\n\n".join([doc.page_content for doc in docs])
        sources = [{"text": doc.page_content, "metadata": doc.metadata} for doc in docs]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        ) from e

    async def generate() -> AsyncGenerator[str, None]:
        try:
            # Create chain
            chain = create_rag_chain(
                model_name=query_request.model or settings.llm_model,
                temperature=query_request.temperature,
            )

            # Stream response
            full_response = ""
            async for chunk in chain.astream(
                {
                    "context": context,
                    "question": query_request.question,
                    "chat_history": chat_history_str,
                }
            ):
                full_response += chunk
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"

            # Save chat messages after streaming completes
            await save_chat_message(db, session.id, "user", query_request.question)
            await save_chat_message(db, session.id, "assistant", full_response)
            await db.commit()

            # Send sources
            yield f"data: {json.dumps({'sources': sources})}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )


@router.post("/{session_id}", response_model=QueryResponse)
@limiter.limit(RATE_LIMIT_QUERY)
async def query(
    request: Request,
    session_id: str,
    query_request: QueryRequest,
    db: AsyncSession = Depends(get_db),
    api_key: APIKey = Depends(get_api_key),
) -> QueryResponse:
    """Query without streaming"""

    logger.info(
        "query_started",
        session_id=session_id,
        question=query_request.question[:50],
    )

    # Validate session
    session = await session_service.get_session_by_str(db, session_id)

    # Get chat history
    chat_messages = await get_chat_history(db, session.id, limit=5)
    chat_history_str = format_chat_history(chat_messages)

    # Contextualize query
    contextualized_question = await contextualize_query(
        query_request.question,
        chat_history_str,
        query_request.model
    )

    if not session:
        logger.warning("query_failed", reason="session_not_found", session_id=session_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    if session.document_count == 0:
        logger.warning("query_failed", reason="no_documents", session_id=session_id)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents in session",
        )

    try:
        # Get relevant documents
        docs = await similarity_search(
            db,
            session.id,
            contextualized_question,
            k=settings.retrieval_top_k,
        )

        # rerank documents
        reranker = get_reranker()
        docs = await reranker.rerank(
            query_request.question,
            docs,
            top_k=settings.rerank_top_k,
        )

        context = "\n\n".join([doc.page_content for doc in docs])

        # Create chain and invoke
        chain = create_rag_chain(
            model_name=query_request.model or settings.llm_model,
            temperature=query_request.temperature,
            language=query_request.language,
        )

        response = await chain.ainvoke(
            {
                "context": context,
                "question": query_request.question,
                "chat_history": "",
            }
        )

        sources = [{"text": doc.page_content, "metadata": doc.metadata} for doc in docs]

        logger.info(
            "query_completed",
            session_id=session_id,
            sources_count=len(sources),
        )

        # save user message
        await save_chat_message(db, session.id, "user", query_request.question)

        # save assistant response
        await save_chat_message(db, session.id, "assistant", response)

        await db.commit()

        return QueryResponse(
            answer=response,
            sources=sources,
            model_used=query_request.model or settings.llm_model,
        )

    except Exception as e:
        logger.error(
            "query_failed",
            session_id=session_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
