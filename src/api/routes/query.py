"""
RAG query endpoints with streaming
"""

import json
import uuid
from collections.abc import AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from src.api.schemas import QueryRequest, QueryResponse
from src.api import session as session_service
from src.core.config import settings
from src.core.exceptions import RAGError
from src.db import get_db
from src.ingestion.pgvector_store import similarity_search
from src.rag.chain import create_rag_chain
from src.rag.prompts import get_prompt
from src.api.dependencies import get_api_key
from src.db.models import APIKey
from starlette.requests import Request
from src.api.rate_limiter import limiter, RATE_LIMIT_QUERY
from src.core.logger import get_logger

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
    
  try:
    # Get relevant documents
    docs = await similarity_search(
      db,
      session.id,
      query_request.question,
      k=settings.retrieval_top_k,
    )
        
    # Format context
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = [
      {"text": doc.page_content, "metadata": doc.metadata}
      for doc in docs
    ]
        
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
            
      prompt = get_prompt(query_request.language)
            
      # Stream response
      full_response = ""
      async for chunk in chain.astream({
        "context": context,
        "question": query_request.question,
        "chat_history": "",
      }):
        full_response += chunk
        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            
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
      query_request.question,
      k=settings.retrieval_top_k,
    )
        
    context = "\n\n".join([doc.page_content for doc in docs])
        
    # Create chain and invoke
    chain = create_rag_chain(
      model_name=query_request.model or settings.llm_model,
      temperature=query_request.temperature,
      language=query_request.language,
    )
        
    response = await chain.ainvoke({
      "context": context,
      "question": query_request.question,
      "chat_history": "",
    })
        
    sources = [
      {"text": doc.page_content, "metadata": doc.metadata}
      for doc in docs
    ]
        
    logger.info(
      "query_completed",
      session_id=session_id,
      sources_count=len(sources),
    )
        
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