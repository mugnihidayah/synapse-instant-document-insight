"""
Document upload and management endpoints
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import Request

from src.api import session as session_service
from src.api.dependencies import get_api_key
from src.api.rate_limiter import RATE_LIMIT_SESSION, RATE_LIMIT_UPLOAD, limiter
from src.api.schemas import (
    DocumentUploadResponse,
    SessionCreate,
    SessionInfo,
)
from src.core.exceptions import DocumentProcessingError, VectorStoreError
from src.core.logger import get_logger
from src.db import get_db
from src.db.models import APIKey
from src.ingestion.chunkers import extract_document_header, split_documents
from src.ingestion.loaders import (
    get_supported_extensions,
    load_document_from_upload,
)
from src.ingestion.pgvector_store import store_documents

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


# SESSION MANAGEMENT
@router.post("/sessions", response_model=SessionCreate)
@limiter.limit(RATE_LIMIT_SESSION)
async def create_session(
    request: Request,
    db: AsyncSession = Depends(get_db),
    api_key: APIKey = Depends(get_api_key),
) -> SessionCreate:
    logger.info("session_create_started")
    session = await session_service.create_session(db, api_key_id=api_key.id)
    logger.info("session_created", session_id=str(session.id))
    return SessionCreate(session_id=str(session.id))


@router.get("/sessions/{session_id}", response_model=SessionInfo)
@limiter.limit(RATE_LIMIT_SESSION)
async def get_session_info(
    request: Request,
    session_id: str,
    db: AsyncSession = Depends(get_db),
    api_key: APIKey = Depends(get_api_key),
) -> SessionInfo:
    """Get session information"""
    session = await session_service.get_session_by_str(db, session_id)
    if session and session.api_key_id != api_key.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="You do not have access to this session"
        )

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    return SessionInfo(
        session_id=str(session.id),
        created_at=session.created_at,
        document_count=session.document_count,
        is_ready=session.document_count > 0,
    )


@router.delete("/sessions/{session_id}")
@limiter.limit(RATE_LIMIT_SESSION)
async def delete_session(
    request: Request,
    session_id: str,
    db: AsyncSession = Depends(get_db),
    api_key: APIKey = Depends(get_api_key),
) -> dict:
    """Delete a session"""
    try:
        uid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session ID format",
        ) from None

    session = await session_service.get_session_for_key(db, uid, api_key.id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found"
        )

    deleted = await session_service.delete_session(db, uid)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    return {"message": f"Session {session_id} deleted"}


# DOCUMENT UPLOAD
@router.post("/upload/{session_id}", response_model=DocumentUploadResponse)
@limiter.limit(RATE_LIMIT_UPLOAD)
async def upload_documents(
    request: Request,
    session_id: str,
    files: Annotated[list[UploadFile], File(...)],
    db: AsyncSession = Depends(get_db),
    api_key: APIKey = Depends(get_api_key),
) -> DocumentUploadResponse:
    """Upload and process documents"""

    logger.info(
        "upload_started",
        session_id=session_id,
        file_count=len(files),
    )

    # Validate session
    session = await session_service.get_session_by_str(db, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found. Create a session first.",
        )

    if session.api_key_id != api_key.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="You do not have access to this session"
        )

    # Validate files
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files uploaded",
        )

    supported = get_supported_extensions()
    for file in files:
        if file.filename is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must have a filename",
            )
        ext = "." + file.filename.split(".")[-1].lower()
        if ext not in supported:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {ext}",
            )

    try:
        # Load documents
        all_documents = []
        for file in files:
            if file.filename is None:
                continue
            docs = load_document_from_upload(file.file, file.filename)
            all_documents.extend(docs)

        if not all_documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No content extracted from files",
            )

        # Chunk documents
        header_chunk = extract_document_header(all_documents)
        chunks = split_documents(all_documents)

        if header_chunk:
            chunks.insert(0, header_chunk)

        # Store in pgvector
        stored_count = await store_documents(db, session.id, chunks)

        logger.info(
            "upload_completed",
            session_id=session_id,
            chunks_created=stored_count,
        )

        return DocumentUploadResponse(
            session_id=session_id,
            document_processed=len(all_documents),
            chunks_created=stored_count,
        )

    except DocumentProcessingError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    except VectorStoreError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.get("/supported-formats")
def get_supported_formats() -> dict:
    """Get supported file formats"""
    return {
        "formats": get_supported_extensions(),
        "description": "Supported document formats for upload",
    }
