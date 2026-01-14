"""
RAG query endpoints with streaming support
"""

import json
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from src.api.schemas import QueryRequest, QueryResponse
from src.api.session import session_manager
from src.core.config import settings
from src.core.exceptions import RAGError
from src.rag.chain import ask_question

router = APIRouter(prefix="/query", tags=["Query"])

async def generate_stream(question: str, messages: list[dict], vectorstore, language: str, model: str | None) -> AsyncGenerator[str, None]:
  """
  Generate streaming response from RAG chain

  Yields SSE formatted data chunks
  """

  try:
    # call RAG chain
    response_generator, sources = ask_question(
      question=question,
      messages=messages,
      vectorstore=vectorstore,
      language=language,
      model_name=model,
    )

    # stream response chunks
    full_response = ""
    for chunk in response_generator:
      full_response += chunk
      # SSE format: data: {json}\n\n
      yield f"data: {json.dumps({"chunk": chunk})}\n\n"

    # send sources at the end
    yield f"data: {json.dumps({"sources": sources})}\n\n"

    # signal completion
    yield f"data: {json.dumps({'done': True, 'full_response': full_response})}\n\n"

  except RAGError as e:
    yield f"data: {json.dumps({'error': str(e)})}\n\n"

@router.post("/stream/{session_id}")
async def query_stream(session_id: str, request: QueryRequest) -> StreamingResponse:
  """
  Query documents with streaming response

  Returns Server Sent Events (SSE) stream

  Args:
    session_id: The session ID with uploaded documents
    request: query parameters

  Returns:
    StreamingResponse with SSE stream
  """

  # get session
  session = session_manager.get_session(session_id)

  if not session:
    raise HTTPException(
      status_code=status.HTTP_404_NOT_FOUND,
      detail=f"Session {session_id} not found"
    )

  if not session.is_ready():
    raise HTTPException(
      status_code=status.HTTP_400_BAD_REQUEST,
      detail="No documents uploaded. Please upload documents first"
    )

  # prepare messages
  messages = [{"role": "user", "content": request.question}]

  # return streaming response
  return StreamingResponse(
    generate_stream(
      question=request.question,
      messages=messages,
      vectorstore=session.vectorstore,
      language=request.language,
      model=request.model,
    ),
    media_type="text/event-stream",
    headers={
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
    },
  )

@router.post("/{session_id}", response_model=QueryResponse)
def query_sync(session_id: str, request: QueryRequest) -> QueryResponse:
  """
  Query documents (non streaming)

  Waits for complete response before returning

  Args:
    session_id: The session ID with uploaded documents
    request: query parameters

  Returns:
    Complete response with answer and sources
  """

  # get session
  session = session_manager.get_session(session_id)

  if not session:
    raise HTTPException(
      status_code=status.HTTP_404_NOT_FOUND,
      detail=f"Session {session_id} not found"
    )

  if not session.is_ready():
    raise HTTPException(
      status_code=status.HTTP_400_BAD_REQUEST,
      detail="No documents uploaded. Please upload documents first"
    )

  try:
    # prepare messages
    messages = [{"role": "user", "content": request.question}]

    # call RAG chain
    response_generator, sources = ask_question(
      question=request.question,
      messages=messages,
      vectorstore=session.vectorstore,
      language=request.language,
      model_name=request.model,
    )

    # consume generator to get full response
    full_response = "".join(response_generator)

    return QueryResponse(
      answer=full_response,
      sources=sources,
      model_used=request.model or settings.llm_model,
    )

  except RAGError as e:
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=str(e)
    )