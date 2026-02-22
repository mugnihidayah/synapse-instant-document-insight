"""Document upload and management endpoints."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import Request

from src.api import session as session_service
from src.api.dependencies import get_api_key
from src.api.ingestion_contract import (
    CODE_FILE_TOO_LARGE,
    CODE_INGESTION_INTERNAL_ERROR,
    CODE_UNSUPPORTED_FORMAT,
    FileIngestionResultData,
    IngestionSummaryData,
    IngestionWarningData,
    UploadFilePayload,
    build_file_result,
    build_ingestion_message,
    build_ingestion_warnings,
    format_ingestion_error,
    log_file_result,
    process_and_store_payloads,
    resolve_session_error_code,
    summarize_file_results,
)
from src.api.ingestion_jobs import schedule_ingestion_job
from src.api.rate_limiter import RATE_LIMIT_SESSION, RATE_LIMIT_UPLOAD, limiter
from src.api.schemas import (
    DocumentUploadResponse,
    FileIngestionResult,
    IngestionSummary,
    IngestionWarning,
    SessionCreate,
    SessionDocumentItem,
    SessionDocumentsResponse,
    SessionInfo,
)
from src.api.usage import record_usage_event
from src.core.config import settings
from src.core.logger import get_logger
from src.db import get_db
from src.db.models import APIKey
from src.ingestion.file_storage import resolve_original_file_path, save_original_file
from src.ingestion.loaders import get_supported_extensions

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


def _to_summary_model(summary: IngestionSummaryData) -> IngestionSummary:
    return IngestionSummary(**summary)


def _to_warning_models(warnings: list[IngestionWarningData]) -> list[IngestionWarning]:
    return [IngestionWarning(**warning) for warning in warnings]


def _to_file_result_models(
    file_results: list[FileIngestionResultData],
) -> list[FileIngestionResult]:
    return [FileIngestionResult(**result) for result in file_results]


def _build_upload_response(
    *,
    session_id: str,
    document_processed: int = 0,
    chunks_created: int = 0,
    files_queued: int = 0,
    ingestion_status: str = "queued",
    message: str = "Documents accepted for processing",
    summary: IngestionSummaryData,
    file_results: list[FileIngestionResultData],
) -> DocumentUploadResponse:
    return DocumentUploadResponse(
        session_id=session_id,
        document_processed=document_processed,
        chunks_created=chunks_created,
        files_queued=files_queued,
        ingestion_status=ingestion_status,
        message=message,
        summary=_to_summary_model(summary),
        file_results=_to_file_result_models(file_results),
    )


@router.post("/sessions", response_model=SessionCreate)
@limiter.limit(RATE_LIMIT_SESSION)
async def create_session(
    request: Request,
    db: AsyncSession = Depends(get_db),
    api_key: APIKey = Depends(get_api_key),
) -> SessionCreate:
    logger.info("session_create_started")
    session = await session_service.create_session(db, api_key_id=api_key.id)
    await record_usage_event(db, api_key.id, "session_create", session_id=session.id)
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
    """Get session information."""
    session = await session_service.get_session_by_str(db, session_id)
    if session and session.api_key_id != api_key.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this session",
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
        is_ready=session.document_count > 0
        and session.ingestion_status in {"ready", "ready_with_warnings"},
        ingestion_status=session.ingestion_status,
        ingestion_error=session.ingestion_error,
        ingestion_summary=_to_summary_model(session_service.get_ingestion_summary(session)),
        ingestion_warnings=_to_warning_models(session_service.get_ingestion_warnings(session)),
        ingestion_started_at=session.ingestion_started_at,
        ingestion_completed_at=session.ingestion_completed_at,
    )


@router.get("/sessions/{session_id}/documents", response_model=SessionDocumentsResponse)
@limiter.limit(RATE_LIMIT_SESSION)
async def list_session_documents(
    request: Request,
    session_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    source: str | None = Query(default=None),
    search: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    api_key: APIKey = Depends(get_api_key),
) -> SessionDocumentsResponse:
    """List document chunks in a session with pagination and optional filtering."""
    session = await session_service.get_session_by_str(db, session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if session.api_key_id != api_key.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this session",
        )

    where_parts = ["session_id = :session_id"]
    params: dict[str, object] = {"session_id": str(session.id)}

    if source:
        where_parts.append("metadata->>'source' = :source")
        params["source"] = source

    if search:
        where_parts.append("content ILIKE :search")
        params["search"] = f"%{search}%"

    where_clause = " AND ".join(where_parts)
    offset = (page - 1) * page_size

    count_stmt = text(f"SELECT COUNT(*) FROM documents WHERE {where_clause}")
    total = int((await db.execute(count_stmt, params)).scalar_one() or 0)

    list_stmt = text(
        f"""
        SELECT id, content, metadata
        FROM documents
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT :limit OFFSET :offset
        """
    )

    rows = (
        await db.execute(
            list_stmt,
            {
                **params,
                "limit": page_size,
                "offset": offset,
            },
        )
    ).fetchall()

    items = [
        SessionDocumentItem(
            chunk_id=str(row.id),
            document_id=(row.metadata or {}).get("document_id"),
            source=(row.metadata or {}).get("source"),
            page=(row.metadata or {}).get("page"),
            section=(row.metadata or {}).get("section"),
            chunk_type=(row.metadata or {}).get("chunk_type"),
            preview=" ".join(row.content.split())[:280],
        )
        for row in rows
    ]

    return SessionDocumentsResponse(
        session_id=session_id,
        total=total,
        page=page,
        page_size=page_size,
        items=items,
    )


@router.get("/{document_id}/file")
@limiter.limit(RATE_LIMIT_SESSION)
async def get_document_file(
    request: Request,
    document_id: str,
    db: AsyncSession = Depends(get_db),
    api_key: APIKey = Depends(get_api_key),
) -> FileResponse:
    """Serve original uploaded file for a logical document_id."""
    stmt = text(
        """
        SELECT session_id,
               metadata->>'file_path' AS file_path,
               COALESCE(metadata->>'original_filename', metadata->>'source', 'document.bin')
                 AS original_filename,
               COALESCE(metadata->>'mime_type', 'application/octet-stream') AS mime_type
        FROM documents
        WHERE metadata->>'document_id' = :document_id
        ORDER BY created_at ASC
        LIMIT 1
        """
    )
    row = (await db.execute(stmt, {"document_id": document_id})).first()
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    try:
        session_uuid = (
            row.session_id
            if isinstance(row.session_id, uuid.UUID)
            else uuid.UUID(str(row.session_id))
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        ) from None

    session = await session_service.get_session(db, session_uuid)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    if session.api_key_id != api_key.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this document",
        )

    file_path = row.file_path
    if not file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Original file for document {document_id} not found",
        )

    try:
        resolved_path = resolve_original_file_path(file_path)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Original file for document {document_id} not found",
        ) from None

    if not resolved_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Original file for document {document_id} not found",
        )

    original_filename = str(row.original_filename or f"{document_id}.bin")
    original_filename = original_filename.replace('"', "")

    return FileResponse(
        path=str(resolved_path),
        media_type=str(row.mime_type or "application/octet-stream"),
        headers={"Content-Disposition": f'inline; filename="{original_filename}"'},
    )


@router.delete("/sessions/{session_id}")
@limiter.limit(RATE_LIMIT_SESSION)
async def delete_session(
    request: Request,
    session_id: str,
    db: AsyncSession = Depends(get_db),
    api_key: APIKey = Depends(get_api_key),
) -> dict:
    """Delete a session."""
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
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    deleted = await session_service.delete_session(db, uid)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    await record_usage_event(db, api_key.id, "session_delete", session_id=uid)
    return {"message": f"Session {session_id} deleted"}


async def _process_documents_sync(
    db: AsyncSession,
    session_id: uuid.UUID,
    payloads: list[UploadFilePayload],
    total_files: int,
    initial_file_results: list[FileIngestionResultData],
    *,
    enable_ocr: bool,
    extract_tables: bool,
) -> tuple[
    int,
    int,
    str,
    str,
    IngestionSummaryData,
    list[FileIngestionResultData],
    list[IngestionWarningData],
    str | None,
]:
    """Synchronous upload path used when async_mode=false."""
    result = await process_and_store_payloads(
        db=db,
        session_id=session_id,
        payloads=payloads,
        total_files=total_files,
        enable_ocr=enable_ocr,
        extract_tables=extract_tables,
        logger=logger,
        initial_file_results=initial_file_results,
    )
    return (
        result.document_processed,
        result.chunks_created,
        result.ingestion_status,
        result.message,
        result.summary,
        result.file_results,
        result.warnings,
        result.error_code,
    )


@router.post("/upload/{session_id}", response_model=DocumentUploadResponse)
@limiter.limit(RATE_LIMIT_UPLOAD)
async def upload_documents(
    request: Request,
    session_id: str,
    files: Annotated[list[UploadFile], File(...)],
    async_mode: bool = Query(default=settings.ingestion_async_default),
    enable_ocr: bool | None = Query(default=None),
    extract_tables: bool | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    api_key: APIKey = Depends(get_api_key),
) -> DocumentUploadResponse:
    """Upload and process documents (async by default)."""
    logger.info(
        "upload_started",
        session_id=session_id,
        file_count=len(files),
        async_mode=async_mode,
    )

    session = await session_service.get_session_by_str(db, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found. Create a session first.",
        )

    if session.api_key_id != api_key.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this session",
        )

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files uploaded",
        )

    supported = set(get_supported_extensions())
    max_bytes = settings.max_upload_file_size_mb * 1024 * 1024

    payloads: list[UploadFilePayload] = []
    initial_file_results: list[FileIngestionResultData] = []
    for file in files:
        mime_type = file.content_type or "application/octet-stream"
        if file.filename is None:
            item = build_file_result(
                filename="unnamed",
                mime_type=mime_type,
                status="failed",
                error_code=CODE_INGESTION_INTERNAL_ERROR,
                severity="error",
                message="File must have a filename",
            )
            initial_file_results.append(item)
            log_file_result(logger, session_id=str(session.id), file_result=item)
            continue

        ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""
        if ext not in supported:
            item = build_file_result(
                filename=file.filename,
                mime_type=mime_type,
                status="failed",
                error_code=CODE_UNSUPPORTED_FORMAT,
                severity="error",
                message=f"Unsupported file type: {ext or 'unknown'}",
            )
            initial_file_results.append(item)
            log_file_result(logger, session_id=str(session.id), file_result=item)
            continue

        content = await file.read()
        if len(content) > max_bytes:
            item = build_file_result(
                filename=file.filename,
                mime_type=mime_type,
                status="failed",
                error_code=CODE_FILE_TOO_LARGE,
                severity="error",
                message=(
                    f"File {file.filename} exceeds {settings.max_upload_file_size_mb}MB limit"
                ),
            )
            initial_file_results.append(item)
            log_file_result(logger, session_id=str(session.id), file_result=item)
            continue

        try:
            stored_file = save_original_file(
                session_id=str(session.id),
                filename=file.filename,
                content=content,
            )
        except Exception as exc:
            logger.error("upload_file_persist_failed", filename=file.filename, error=str(exc))
            item = build_file_result(
                filename=file.filename,
                mime_type=mime_type,
                status="failed",
                error_code=CODE_INGESTION_INTERNAL_ERROR,
                severity="error",
                message=f"Failed to persist uploaded file {file.filename}",
            )
            initial_file_results.append(item)
            log_file_result(logger, session_id=str(session.id), file_result=item)
            continue

        payloads.append(
            UploadFilePayload(
                filename=stored_file.original_filename,
                content=content,
                document_id=stored_file.document_id,
                file_path=stored_file.file_path,
                mime_type=stored_file.mime_type,
                file_size_bytes=stored_file.file_size_bytes,
            )
        )

    total_files = len(files)
    queued_summary = summarize_file_results(
        total_files=total_files,
        file_results=initial_file_results,
    )
    queued_warnings = build_ingestion_warnings(initial_file_results)

    if not payloads:
        error_code = resolve_session_error_code(
            summary=queued_summary,
            file_results=initial_file_results,
        )
        message = build_ingestion_message("failed", error_code)
        await session_service.set_ingestion_status(
            db,
            session.id,
            "failed",
            error=format_ingestion_error(error_code, message),
            summary=queued_summary,
            warnings=queued_warnings,
            file_results=initial_file_results,
            error_code=error_code,
        )
        return _build_upload_response(
            session_id=session_id,
            document_processed=0,
            chunks_created=0,
            files_queued=0,
            ingestion_status="failed",
            message=message,
            summary=queued_summary,
            file_results=initial_file_results,
        )

    ocr_enabled = settings.enable_ocr if enable_ocr is None else enable_ocr
    table_enabled = settings.enable_table_extraction if extract_tables is None else extract_tables

    await session_service.set_ingestion_status(
        db,
        session.id,
        "queued",
        summary=queued_summary,
        warnings=queued_warnings,
        file_results=initial_file_results,
        error_code=None,
    )

    if async_mode:
        job_id = schedule_ingestion_job(
            session.id,
            payloads,
            total_files=total_files,
            initial_file_results=initial_file_results,
            enable_ocr=ocr_enabled,
            extract_tables=table_enabled,
        )

        await record_usage_event(
            db,
            api_key.id,
            "upload_queued",
            session_id=session.id,
            metadata={
                "job_id": job_id,
                "total_files": total_files,
                "files": len(payloads),
                "ocr_enabled": ocr_enabled,
                "table_extraction_enabled": table_enabled,
            },
        )

        return _build_upload_response(
            session_id=session_id,
            files_queued=len(payloads),
            ingestion_status="queued",
            message=(
                "Documents accepted for background processing. "
                f"Use /documents/sessions/{session_id} to track status. job_id={job_id}"
            ),
            summary=queued_summary,
            file_results=initial_file_results,
        )

    try:
        await session_service.set_ingestion_status(
            db,
            session.id,
            "processing",
            summary=queued_summary,
            warnings=queued_warnings,
            file_results=initial_file_results,
            error_code=None,
        )
        (
            docs_count,
            chunk_count,
            ingestion_status,
            message,
            summary,
            file_results,
            warnings,
            error_code,
        ) = await _process_documents_sync(
            db,
            session.id,
            payloads,
            total_files=total_files,
            initial_file_results=initial_file_results,
            enable_ocr=ocr_enabled,
            extract_tables=table_enabled,
        )

        await session_service.set_ingestion_status(
            db,
            session.id,
            ingestion_status,
            error=(
                format_ingestion_error(error_code, message)
                if ingestion_status == "failed"
                else None
            ),
            summary=summary,
            warnings=warnings,
            file_results=file_results,
            error_code=error_code,
        )
        if ingestion_status in {"ready", "ready_with_warnings"}:
            await record_usage_event(
                db,
                api_key.id,
                "upload",
                session_id=session.id,
                metadata={
                    "files": len(payloads),
                    "documents": docs_count,
                    "chunks": chunk_count,
                    "ocr_enabled": ocr_enabled,
                    "table_extraction_enabled": table_enabled,
                },
            )

        logger.info(
            "upload_completed",
            session_id=session_id,
            chunks_created=chunk_count,
            ingestion_status=ingestion_status,
        )

        return _build_upload_response(
            session_id=session_id,
            document_processed=docs_count,
            chunks_created=chunk_count,
            files_queued=0,
            ingestion_status=ingestion_status,
            message=message,
            summary=summary,
            file_results=file_results,
        )
    except Exception as exc:
        logger.error("upload_processing_failed", session_id=session_id, error=str(exc))
        summary = summarize_file_results(total_files=total_files, file_results=initial_file_results)
        warnings = build_ingestion_warnings(initial_file_results)
        error_code = CODE_INGESTION_INTERNAL_ERROR
        message = build_ingestion_message("failed", error_code)
        await session_service.set_ingestion_status(
            db,
            session.id,
            "failed",
            error=format_ingestion_error(error_code, message),
            summary=summary,
            warnings=warnings,
            file_results=initial_file_results,
            error_code=error_code,
        )
        return _build_upload_response(
            session_id=session_id,
            document_processed=0,
            chunks_created=0,
            files_queued=0,
            ingestion_status="failed",
            message=message,
            summary=summary,
            file_results=initial_file_results,
        )


@router.get("/supported-formats")
def get_supported_formats() -> dict:
    """Get supported file formats."""
    return {
        "formats": get_supported_extensions(),
        "description": "Supported document formats for upload",
    }
