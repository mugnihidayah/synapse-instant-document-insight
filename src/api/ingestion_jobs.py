"""Background ingestion jobs for async document processing."""

import asyncio
import io
import uuid
from typing import TypedDict

from src.api import session as session_service
from src.api.usage import record_usage_event
from src.core.config import settings
from src.core.exceptions import DocumentProcessingError
from src.core.logger import get_logger
from src.db.connection import get_db_context
from src.ingestion.chunkers import extract_document_header, split_documents
from src.ingestion.loaders import load_document_from_upload
from src.ingestion.pgvector_store import store_documents

logger = get_logger(__name__)


class UploadFilePayload(TypedDict):
    filename: str
    content: bytes
    document_id: str
    file_path: str
    mime_type: str
    file_size_bytes: int


_jobs: dict[str, asyncio.Task] = {}


def schedule_ingestion_job(
    session_id: uuid.UUID,
    payloads: list[UploadFilePayload],
    *,
    enable_ocr: bool | None = None,
    extract_tables: bool | None = None,
) -> str:
    """Create and track a background ingestion task."""
    job_id = str(uuid.uuid4())
    task = asyncio.create_task(
        _run_ingestion_job(
            session_id,
            payloads,
            enable_ocr=settings.enable_ocr if enable_ocr is None else enable_ocr,
            extract_tables=(
                settings.enable_table_extraction if extract_tables is None else extract_tables
            ),
        )
    )
    _jobs[job_id] = task

    def _cleanup(_: asyncio.Task) -> None:
        _jobs.pop(job_id, None)

    task.add_done_callback(_cleanup)
    return job_id


async def _run_ingestion_job(
    session_id: uuid.UUID,
    payloads: list[UploadFilePayload],
    *,
    enable_ocr: bool,
    extract_tables: bool,
) -> None:
    api_key_id: uuid.UUID | None = None

    try:
        async with get_db_context() as db:
            session = await session_service.set_ingestion_status(db, session_id, "processing")
            if not session:
                logger.warning("ingestion_session_missing", session_id=str(session_id))
                return
            api_key_id = session.api_key_id

            all_documents = []
            for payload in payloads:
                document_metadata = {
                    "document_id": payload["document_id"],
                    "original_filename": payload["filename"],
                    "file_path": payload["file_path"],
                    "mime_type": payload["mime_type"],
                    "file_size_bytes": payload["file_size_bytes"],
                }
                docs = load_document_from_upload(
                    io.BytesIO(payload["content"]),
                    payload["filename"],
                    enable_ocr=enable_ocr,
                    extract_tables=extract_tables,
                    document_metadata=document_metadata,
                )
                all_documents.extend(docs)

            if not all_documents:
                raise DocumentProcessingError("No content extracted from uploaded files")

            header_chunk = extract_document_header(all_documents)
            chunks = split_documents(all_documents)
            if header_chunk:
                chunks.insert(0, header_chunk)

            stored_count = await store_documents(db, session_id, chunks)
            await session_service.set_ingestion_status(db, session_id, "ready")

            if api_key_id:
                await record_usage_event(
                    db,
                    api_key_id,
                    "upload",
                    session_id=session_id,
                    metadata={
                        "files": len(payloads),
                        "chunks": stored_count,
                        "ocr_enabled": enable_ocr,
                        "table_extraction_enabled": extract_tables,
                    },
                )

            logger.info(
                "ingestion_job_completed",
                session_id=str(session_id),
                files=len(payloads),
                chunks=stored_count,
            )

    except Exception as exc:
        logger.error("ingestion_job_failed", session_id=str(session_id), error=str(exc))
        async with get_db_context() as db:
            await session_service.set_ingestion_status(db, session_id, "failed", error=str(exc))
