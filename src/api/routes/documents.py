"""
Document uplaod and management endpoints
"""

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from src.api.schemas import (
  DocumentUploadResponse,
  SessionCreate,
  SessionInfo,
)
from src.api.session import session_manager
from src.core.exceptions import DocumentProcessingError, VectorStoreError
from src.ingestion.loaders import (
  get_supported_extensions,
  load_document_from_upload,
)
from src.ingestion.chunkers import split_documents
from src.ingestion.vectorstore import create_vectorstore

router = APIRouter(prefix="/documents", tags=["Documents"])

# SESSION MANAGEMENT
@router.post("/sessions", response_model=SessionCreate)
def create_session() -> SessionCreate:
  """
  Create a new session for document upload

  Returns:
    Session ID to use for subsequent requests
  """

  session = session_manager.create_session()
  return SessionCreate(session_id=session.id)

@router.get("/sessions/{session_id}", response_model=SessionInfo)
def get_session(session_id: str) -> SessionInfo:
  """
  Get information about a session

  Args:
    session_id: The session ID

  Returns:
    Session information including document count
  """

  session = session_manager.get_session(session_id)

  if not session:
    raise HTTPException(
      status_code=status.HTTP_404_NOT_FOUND,
      detail=f"Session {session_id} not found"
    )

  return SessionInfo(
    session_id=session.id,
    created_at=session.created_at,
    document_count=session.document_count,
    is_ready=session.is_ready(),
  )

@router.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
  """
  Delete a session and its associated data

  Args:
    session_id: The session ID to delete
  """

  deleted = session_manager.delete_session(session_id)

  if not deleted:
    raise HTTPException(
      status_code=status.HTTP_404_NOT_FOUND,
      detail=f"Session {session_id} not found"
    )

  return {"message": f"Session {session_id} deleted"}

# DOCUMENT UPLOAD
@router.post("/upload/{session_id}", response_model=DocumentUploadResponse)
async def upload_documents(session_id: str, files: list[UploadFile] = File(...)) -> DocumentUploadResponse:
  """
  Upload and process documents

  Args:
    session_id: The session ID
    files: List of uploaded files to upload (PDF, DOCX, TXT)

  Returns:
    Processing result including chunk count
  """

  # validate session
  session = session_manager.get_session(session_id)

  if not session:
    raise HTTPException(
      status_code=status.HTTP_404_NOT_FOUND,
      detail=f"Session {session_id} not found. Create a session first"
    )

  # validate files
  if not files:
    raise HTTPException(
      status_code=status.HTTP_400_BAD_REQUEST,
      detail="No files uploaded"
    )

  supported = get_supported_extensions()
  for file in files:
    ext = "." + file.filename.split(".")[-1].lower()
    if ext not in supported:
      raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Unsupported file type: {ext}. Supported types: {supported}"
      )

  try:
    # load documents
    all_documents = []
    for file in files:
      docs = load_document_from_upload(file.file, file.filename)
      all_documents.extend(docs)

    if not all_documents:
      raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="No content extracted from files"
      )

    # chunk documents
    chunks = split_documents(all_documents)

    # create vectorstore
    vectorstore = create_vectorstore(chunks)

    # update session
    session_manager.update_session(
      session_id=session_id,
      vectorstore=vectorstore,
      document_count=len(all_documents),
    )

    return DocumentUploadResponse(
      session_id=session_id,
      document_processed=len(all_documents),
      chunks_created=len(chunks),
    )

  except DocumentProcessingError as e:
    raise HTTPException(
      status_code=status.HTTP_400_BAD_REQUEST,
      detail=str(e)
    )

  except VectorStoreError as e:
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=str(e)
    )

  except VectorStoreError as e:
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=str(e)
    )

@router.get("/supported-formats")
def get_supported_formats() -> dict:
  """
  Get list of supported file formats

  Returns:
    List of supported file extensions
  """

  return {
    "formats": get_supported_extentions(),
    "description": "Supported document formats for upload",
  }