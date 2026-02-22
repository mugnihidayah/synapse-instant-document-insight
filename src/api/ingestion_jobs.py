"""Background ingestion jobs for async document processing."""

import asyncio
import uuid

from src.api import session as session_service
from src.api.ingestion_contract import (
    CODE_INGESTION_INTERNAL_ERROR,
    FileIngestionResultData,
    UploadFilePayload,
    build_ingestion_message,
    build_ingestion_warnings,
    format_ingestion_error,
    process_and_store_payloads,
    summarize_file_results,
)
from src.api.usage import record_usage_event
from src.core.config import settings
from src.core.logger import get_logger
from src.db.connection import get_db_context

logger = get_logger(__name__)


_jobs: dict[str, asyncio.Task] = {}


def schedule_ingestion_job(
    session_id: uuid.UUID,
    payloads: list[UploadFilePayload],
    *,
    total_files: int,
    initial_file_results: list[FileIngestionResultData] | None = None,
    enable_ocr: bool | None = None,
    extract_tables: bool | None = None,
) -> str:
    """Create and track a background ingestion task."""
    job_id = str(uuid.uuid4())
    task = asyncio.create_task(
        _run_ingestion_job(
            session_id,
            payloads,
            total_files=total_files,
            initial_file_results=initial_file_results or [],
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
    total_files: int,
    initial_file_results: list[FileIngestionResultData],
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
            await session_service.set_ingestion_status(
                db,
                session_id,
                result.ingestion_status,
                error=(
                    format_ingestion_error(result.error_code, result.message)
                    if result.ingestion_status == "failed"
                    else None
                ),
                summary=result.summary,  # type: ignore
                warnings=result.warnings,  # type: ignore
                file_results=result.file_results,  # type: ignore
                error_code=result.error_code,
            )

            if api_key_id and result.ingestion_status in {"ready", "ready_with_warnings"}:
                await record_usage_event(
                    db,
                    api_key_id,
                    "upload",
                    session_id=session_id,
                    metadata={
                        "files": result.summary["total_files"],
                        "documents": result.document_processed,
                        "chunks": result.chunks_created,
                        "ocr_enabled": enable_ocr,
                        "table_extraction_enabled": extract_tables,
                    },
                )

            logger.info(
                "ingestion_job_completed",
                session_id=str(session_id),
                files=result.summary["total_files"],
                chunks=result.chunks_created,
                ingestion_status=result.ingestion_status,
            )

    except Exception as exc:
        logger.error("ingestion_job_failed", session_id=str(session_id), error=str(exc))
        async with get_db_context() as db:
            summary = summarize_file_results(
                total_files=total_files,
                file_results=initial_file_results,
            )
            warnings = build_ingestion_warnings(initial_file_results)
            message = build_ingestion_message("failed", CODE_INGESTION_INTERNAL_ERROR)
            await session_service.set_ingestion_status(
                db,
                session_id,
                "failed",
                error=format_ingestion_error(CODE_INGESTION_INTERNAL_ERROR, message),
                summary=summary,  # type: ignore
                warnings=warnings,  # type: ignore
                file_results=initial_file_results,  # type: ignore
                error_code=CODE_INGESTION_INTERNAL_ERROR,
            )
