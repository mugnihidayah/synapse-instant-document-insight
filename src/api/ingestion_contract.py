"""Structured ingestion contract helpers shared by upload and session APIs."""

from __future__ import annotations

import io
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langchain_core.documents import Document
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import DocumentProcessingError
from src.ingestion.chunkers import extract_document_header, split_documents
from src.ingestion.loaders import load_document_from_upload

FileStatus = Literal["processed", "warning", "failed"]
SeverityLevel = Literal["warning", "error"]

FILE_STATUS_PROCESSED: FileStatus = "processed"
FILE_STATUS_WARNING: FileStatus = "warning"
FILE_STATUS_FAILED: FileStatus = "failed"

SEVERITY_WARNING: SeverityLevel = "warning"
SEVERITY_ERROR: SeverityLevel = "error"

INGESTION_STATUS_READY_WITH_WARNINGS = "ready_with_warnings"

CODE_OCR_NO_TEXT = "OCR_NO_TEXT"
CODE_UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
CODE_FILE_TOO_LARGE = "FILE_TOO_LARGE"
CODE_OCR_ENGINE_UNAVAILABLE = "OCR_ENGINE_UNAVAILABLE"
CODE_INGESTION_INTERNAL_ERROR = "INGESTION_INTERNAL_ERROR"
CODE_NO_USABLE_DOCUMENTS = "NO_USABLE_DOCUMENTS"


class UploadFilePayload(TypedDict):
    filename: str
    content: bytes
    document_id: str
    file_path: str
    mime_type: str
    file_size_bytes: int


class IngestionSummaryData(TypedDict):
    total_files: int
    processed_files: int
    warning_files: int
    failed_files: int


class FileIngestionResultData(TypedDict):
    filename: str
    mime_type: str
    status: FileStatus
    error_code: str | None
    severity: SeverityLevel | None
    message: str | None
    document_id: str | None
    chunks_created: int | None


class IngestionWarningData(TypedDict):
    code: str
    message: str
    filename: str | None


@dataclass
class IngestionExecutionResult:
    """Combined ingestion result for upload response + session tracking."""

    document_processed: int
    chunks_created: int
    ingestion_status: str
    message: str
    summary: IngestionSummaryData
    file_results: list[FileIngestionResultData]
    warnings: list[IngestionWarningData]
    error_code: str | None = None


def empty_ingestion_summary(total_files: int = 0) -> IngestionSummaryData:
    return IngestionSummaryData(
        total_files=total_files,
        processed_files=0,
        warning_files=0,
        failed_files=0,
    )


def normalize_ingestion_summary(summary: Mapping[str, Any] | None) -> IngestionSummaryData:
    if summary is None:
        return empty_ingestion_summary()
    return IngestionSummaryData(
        total_files=max(int(summary.get("total_files", 0) or 0), 0),
        processed_files=max(int(summary.get("processed_files", 0) or 0), 0),
        warning_files=max(int(summary.get("warning_files", 0) or 0), 0),
        failed_files=max(int(summary.get("failed_files", 0) or 0), 0),
    )


def normalize_ingestion_warnings(
    warnings: Sequence[Mapping[str, Any]] | None,
) -> list[IngestionWarningData]:
    if warnings is None:
        return []

    normalized: list[IngestionWarningData] = []
    for warning in warnings:
        code = str(warning.get("code") or "").strip()
        message = str(warning.get("message") or "").strip()
        if not code or not message:
            continue
        filename = warning.get("filename")
        normalized.append(
            IngestionWarningData(
                code=code,
                message=message,
                filename=str(filename) if filename else None,
            )
        )
    return normalized


def normalize_file_results(
    file_results: Sequence[Mapping[str, Any]] | None,
) -> list[FileIngestionResultData]:
    if file_results is None:
        return []

    normalized: list[FileIngestionResultData] = []
    for result in file_results:
        status_raw = str(result.get("status") or "").strip()
        if status_raw == FILE_STATUS_PROCESSED:
            status = FILE_STATUS_PROCESSED
        elif status_raw == FILE_STATUS_WARNING:
            status = FILE_STATUS_WARNING
        elif status_raw == FILE_STATUS_FAILED:
            status = FILE_STATUS_FAILED
        else:
            continue

        severity_raw = str(result.get("severity") or "").strip()
        severity: SeverityLevel | None = None
        if severity_raw == SEVERITY_WARNING:
            severity = SEVERITY_WARNING
        elif severity_raw == SEVERITY_ERROR:
            severity = SEVERITY_ERROR

        chunks_created_raw = result.get("chunks_created")
        chunks_created: int | None = None
        if chunks_created_raw is not None:
            try:
                chunks_created = int(chunks_created_raw)
            except (TypeError, ValueError):
                chunks_created = None

        normalized.append(
            FileIngestionResultData(
                filename=str(result.get("filename") or ""),
                mime_type=str(result.get("mime_type") or "application/octet-stream"),
                status=status,
                error_code=str(result.get("error_code")) if result.get("error_code") else None,
                severity=severity,
                message=str(result.get("message")) if result.get("message") else None,
                document_id=str(result.get("document_id")) if result.get("document_id") else None,
                chunks_created=chunks_created,
            )
        )
    return normalized


def build_file_result(
    *,
    filename: str,
    mime_type: str,
    status: FileStatus,
    document_id: str | None = None,
    error_code: str | None = None,
    severity: SeverityLevel | None = None,
    message: str | None = None,
    chunks_created: int | None = None,
) -> FileIngestionResultData:
    return FileIngestionResultData(
        filename=filename,
        mime_type=mime_type or "application/octet-stream",
        status=status,
        error_code=error_code,
        severity=severity,
        message=message,
        document_id=document_id,
        chunks_created=chunks_created,
    )


def log_file_result(
    logger: Any,
    *,
    session_id: str,
    file_result: FileIngestionResultData,
) -> None:
    logger.info(
        "ingestion_file_result",
        session_id=session_id,
        filename=file_result["filename"],
        status=file_result["status"],
        error_code=file_result["error_code"],
    )


def classify_document_error(
    error: DocumentProcessingError | Exception,
) -> tuple[FileStatus, str, SeverityLevel, str]:
    if isinstance(error, DocumentProcessingError):
        details = error.details if isinstance(error.details, dict) else {}
        message = str(error.message).strip() or "Document processing failed"

        code = details.get("error_code")
        severity = details.get("severity")
        if (
            isinstance(code, str)
            and isinstance(severity, str)
            and severity in {SEVERITY_WARNING, SEVERITY_ERROR}
        ):
            normalized_severity: SeverityLevel = (
                SEVERITY_WARNING if severity == SEVERITY_WARNING else SEVERITY_ERROR
            )
            normalized_status: FileStatus = (
                FILE_STATUS_WARNING
                if normalized_severity == SEVERITY_WARNING
                else FILE_STATUS_FAILED
            )
            return normalized_status, code, normalized_severity, message

        lowered = message.lower()
        if "no text could be extracted from image" in lowered:
            return FILE_STATUS_WARNING, CODE_OCR_NO_TEXT, SEVERITY_WARNING, message
        if "unsupported file format" in lowered:
            return FILE_STATUS_FAILED, CODE_UNSUPPORTED_FORMAT, SEVERITY_ERROR, message
        if "ocr" in lowered and "unavailable" in lowered:
            return FILE_STATUS_FAILED, CODE_OCR_ENGINE_UNAVAILABLE, SEVERITY_ERROR, message
        return FILE_STATUS_FAILED, CODE_INGESTION_INTERNAL_ERROR, SEVERITY_ERROR, message

    return (
        FILE_STATUS_FAILED,
        CODE_INGESTION_INTERNAL_ERROR,
        SEVERITY_ERROR,
        "Unhandled ingestion error",
    )


def summarize_file_results(
    *,
    total_files: int,
    file_results: list[FileIngestionResultData],
) -> IngestionSummaryData:
    summary = empty_ingestion_summary(total_files=total_files)

    for item in file_results:
        if item["status"] == FILE_STATUS_PROCESSED:
            summary["processed_files"] += 1
        elif item["status"] == FILE_STATUS_WARNING:
            summary["warning_files"] += 1
        elif item["status"] == FILE_STATUS_FAILED:
            summary["failed_files"] += 1

    return summary


def build_ingestion_warnings(
    file_results: list[FileIngestionResultData],
) -> list[IngestionWarningData]:
    warnings: list[IngestionWarningData] = []
    for item in file_results:
        if item["status"] != FILE_STATUS_WARNING:
            continue
        warnings.append(
            IngestionWarningData(
                code=item["error_code"] or CODE_OCR_NO_TEXT,
                message=item["message"] or "Document produced warning",
                filename=item["filename"] or None,
            )
        )
    return warnings


def resolve_ingestion_status(summary: IngestionSummaryData) -> str:
    if summary["processed_files"] == 0:
        return "failed"
    if summary["warning_files"] > 0:
        return INGESTION_STATUS_READY_WITH_WARNINGS
    return "ready"


def resolve_session_error_code(
    *,
    summary: IngestionSummaryData,
    file_results: list[FileIngestionResultData],
) -> str | None:
    if summary["processed_files"] > 0:
        return None

    for item in file_results:
        if item["status"] == FILE_STATUS_FAILED and item["error_code"]:
            return item["error_code"]

    return CODE_NO_USABLE_DOCUMENTS


def build_ingestion_message(status: str, error_code: str | None = None) -> str:
    if status == "queued":
        return "Documents accepted for background processing"
    if status == "ready":
        return "Documents processed successfully"
    if status == INGESTION_STATUS_READY_WITH_WARNINGS:
        return "Documents processed with warnings"
    if error_code == CODE_NO_USABLE_DOCUMENTS:
        return "No usable documents were produced from uploaded files"
    if error_code == CODE_OCR_ENGINE_UNAVAILABLE:
        return "OCR engine is unavailable"
    if error_code == CODE_UNSUPPORTED_FORMAT:
        return "Uploaded files are not in a supported format"
    if error_code == CODE_FILE_TOO_LARGE:
        return "Uploaded files exceed size limit"
    return "Document ingestion failed"


def format_ingestion_error(error_code: str | None, message: str | None) -> str | None:
    if not message:
        return None
    if not error_code:
        return message
    return f"{error_code}: {message}"


def _assign_chunk_counts(
    file_results: list[FileIngestionResultData], chunks: list[Document]
) -> None:
    chunk_counts: Counter[str] = Counter()
    for chunk in chunks:
        doc_id = (chunk.metadata or {}).get("document_id")
        if doc_id:
            chunk_counts[str(doc_id)] += 1

    for item in file_results:
        if item["status"] != FILE_STATUS_PROCESSED:
            continue
        doc_id = item.get("document_id")
        item["chunks_created"] = chunk_counts.get(str(doc_id), 0) if doc_id else 0


async def _store_documents(
    db: AsyncSession,
    session_id: uuid.UUID,
    chunks: list[Document],
) -> int:
    """Lazy import wrapper to keep optional DB dependencies out of unit-test import path."""
    from src.ingestion.pgvector_store import store_documents

    return await store_documents(db, session_id, chunks)


def process_upload_payloads(
    *,
    session_id: uuid.UUID,
    payloads: list[UploadFilePayload],
    enable_ocr: bool,
    extract_tables: bool,
    logger: Any,
) -> tuple[list[Document], list[FileIngestionResultData]]:
    all_documents: list[Document] = []
    file_results: list[FileIngestionResultData] = []

    for payload in payloads:
        try:
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
            item = build_file_result(
                filename=payload["filename"],
                mime_type=payload["mime_type"],
                status=FILE_STATUS_PROCESSED,
                document_id=payload["document_id"],
            )
            file_results.append(item)
            log_file_result(logger, session_id=str(session_id), file_result=item)
        except DocumentProcessingError as exc:
            status, error_code, severity, message = classify_document_error(exc)
            item = build_file_result(
                filename=payload["filename"],
                mime_type=payload["mime_type"],
                status=status,
                document_id=payload["document_id"],
                error_code=error_code,
                severity=severity,
                message=message,
            )
            file_results.append(item)
            log_file_result(logger, session_id=str(session_id), file_result=item)
        except Exception:
            item = build_file_result(
                filename=payload["filename"],
                mime_type=payload["mime_type"],
                status=FILE_STATUS_FAILED,
                document_id=payload["document_id"],
                error_code=CODE_INGESTION_INTERNAL_ERROR,
                severity=SEVERITY_ERROR,
                message="Unhandled ingestion error",
            )
            file_results.append(item)
            log_file_result(logger, session_id=str(session_id), file_result=item)

    return all_documents, file_results


async def process_and_store_payloads(
    *,
    db: AsyncSession,
    session_id: uuid.UUID,
    payloads: list[UploadFilePayload],
    total_files: int,
    enable_ocr: bool,
    extract_tables: bool,
    logger: Any,
    initial_file_results: list[FileIngestionResultData] | None = None,
) -> IngestionExecutionResult:
    base_results = list(initial_file_results or [])
    documents, processed_results = process_upload_payloads(
        session_id=session_id,
        payloads=payloads,
        enable_ocr=enable_ocr,
        extract_tables=extract_tables,
        logger=logger,
    )
    file_results = [*base_results, *processed_results]
    summary = summarize_file_results(total_files=total_files, file_results=file_results)
    warnings = build_ingestion_warnings(file_results)

    if not documents:
        error_code = resolve_session_error_code(summary=summary, file_results=file_results)
        return IngestionExecutionResult(
            document_processed=0,
            chunks_created=0,
            ingestion_status="failed",
            message=build_ingestion_message("failed", error_code),
            summary=summary,
            file_results=file_results,
            warnings=warnings,
            error_code=error_code,
        )

    header_chunk = extract_document_header(documents)
    chunks = split_documents(documents)
    if header_chunk:
        chunks.insert(0, header_chunk)

    _assign_chunk_counts(file_results, chunks)

    try:
        stored_count = await _store_documents(db, session_id, chunks)
    except Exception:
        error_code = CODE_INGESTION_INTERNAL_ERROR
        return IngestionExecutionResult(
            document_processed=0,
            chunks_created=0,
            ingestion_status="failed",
            message=build_ingestion_message("failed", error_code),
            summary=summary,
            file_results=file_results,
            warnings=warnings,
            error_code=error_code,
        )

    ingestion_status = resolve_ingestion_status(summary)
    return IngestionExecutionResult(
        document_processed=len(documents),
        chunks_created=stored_count,
        ingestion_status=ingestion_status,
        message=build_ingestion_message(ingestion_status),
        summary=summary,
        file_results=file_results,
        warnings=warnings,
        error_code=None,
    )
