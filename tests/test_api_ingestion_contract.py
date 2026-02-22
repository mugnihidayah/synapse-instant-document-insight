"""Tests for structured ingestion upload/session contract."""

import uuid
from typing import Any

import pytest
from langchain_core.documents import Document

from src.api.ingestion_contract import (
    CODE_NO_USABLE_DOCUMENTS,
    CODE_OCR_ENGINE_UNAVAILABLE,
    CODE_OCR_NO_TEXT,
    INGESTION_STATUS_READY_WITH_WARNINGS,
    UploadFilePayload,
    process_and_store_payloads,
)
from src.core.exceptions import DocumentProcessingError


class _DummyLogger:
    def info(self, *_args: Any, **_kwargs: Any) -> None:
        return


def _make_payload(filename: str, *, document_id: str) -> UploadFilePayload:
    return UploadFilePayload(
        filename=filename,
        content=b"dummy-content",
        document_id=document_id,
        file_path=f"s-1/{document_id}/{filename}",
        mime_type="application/octet-stream",
        file_size_bytes=13,
    )


@pytest.mark.asyncio
async def test_ingestion_pdf_normal_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_store(_db: object, _session_id: uuid.UUID, chunks: list[Document]) -> int:
        return len(chunks)

    def fake_loader(*_args: Any, **_kwargs: Any) -> list[Document]:
        return [Document(page_content="pdf text", metadata={})]

    monkeypatch.setattr("src.api.ingestion_contract.load_document_from_upload", fake_loader)
    monkeypatch.setattr("src.api.ingestion_contract.extract_document_header", lambda _docs: None)
    monkeypatch.setattr(
        "src.api.ingestion_contract.split_documents",
        lambda docs: [Document(page_content="chunk", metadata=dict(docs[0].metadata))],
    )
    monkeypatch.setattr("src.api.ingestion_contract._store_documents", fake_store)

    result = await process_and_store_payloads(
        db=object(),  # type: ignore[arg-type]
        session_id=uuid.uuid4(),
        payloads=[_make_payload("report.pdf", document_id="doc-pdf")],
        total_files=1,
        enable_ocr=True,
        extract_tables=False,
        logger=_DummyLogger(),
        initial_file_results=[],
    )

    assert result.ingestion_status == "ready"
    assert result.summary == {
        "total_files": 1,
        "processed_files": 1,
        "warning_files": 0,
        "failed_files": 0,
    }
    assert result.file_results[0]["status"] == "processed"
    assert result.file_results[0]["error_code"] is None


@pytest.mark.asyncio
async def test_ingestion_image_no_text_failed_with_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_store(_db: object, _session_id: uuid.UUID, _chunks: list[Document]) -> int:
        return 0

    def fake_loader(*_args: Any, **_kwargs: Any) -> list[Document]:
        raise DocumentProcessingError(
            "No text could be extracted from image",
            details={"error_code": CODE_OCR_NO_TEXT, "severity": "warning"},
        )

    monkeypatch.setattr("src.api.ingestion_contract.load_document_from_upload", fake_loader)
    monkeypatch.setattr("src.api.ingestion_contract._store_documents", fake_store)

    result = await process_and_store_payloads(
        db=object(),  # type: ignore[arg-type]
        session_id=uuid.uuid4(),
        payloads=[_make_payload("photo.png", document_id="doc-img")],
        total_files=1,
        enable_ocr=True,
        extract_tables=False,
        logger=_DummyLogger(),
        initial_file_results=[],
    )

    assert result.ingestion_status == "failed"
    assert result.error_code == CODE_NO_USABLE_DOCUMENTS
    assert result.summary == {
        "total_files": 1,
        "processed_files": 0,
        "warning_files": 1,
        "failed_files": 0,
    }
    assert result.file_results[0]["status"] == "warning"
    assert result.file_results[0]["error_code"] == CODE_OCR_NO_TEXT


@pytest.mark.asyncio
async def test_ingestion_mixed_ready_with_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_store(_db: object, _session_id: uuid.UUID, chunks: list[Document]) -> int:
        return len(chunks)

    def fake_loader(_upload: Any, filename: str, **_kwargs: Any) -> list[Document]:
        if filename.endswith(".pdf"):
            return [Document(page_content="pdf text", metadata={})]
        raise DocumentProcessingError(
            "No text could be extracted from image",
            details={"error_code": CODE_OCR_NO_TEXT, "severity": "warning"},
        )

    monkeypatch.setattr("src.api.ingestion_contract.load_document_from_upload", fake_loader)
    monkeypatch.setattr("src.api.ingestion_contract.extract_document_header", lambda _docs: None)
    monkeypatch.setattr(
        "src.api.ingestion_contract.split_documents",
        lambda docs: [Document(page_content="chunk", metadata=dict(docs[0].metadata))],
    )
    monkeypatch.setattr("src.api.ingestion_contract._store_documents", fake_store)

    result = await process_and_store_payloads(
        db=object(),  # type: ignore[arg-type]
        session_id=uuid.uuid4(),
        payloads=[
            _make_payload("report.pdf", document_id="doc-pdf"),
            _make_payload("photo.png", document_id="doc-img"),
        ],
        total_files=2,
        enable_ocr=True,
        extract_tables=False,
        logger=_DummyLogger(),
        initial_file_results=[],
    )

    assert result.ingestion_status == INGESTION_STATUS_READY_WITH_WARNINGS
    assert result.summary == {
        "total_files": 2,
        "processed_files": 1,
        "warning_files": 1,
        "failed_files": 0,
    }
    assert any(item["status"] == "processed" for item in result.file_results)
    assert any(item["error_code"] == CODE_OCR_NO_TEXT for item in result.file_results)


@pytest.mark.asyncio
async def test_ingestion_ocr_engine_down_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_store(_db: object, _session_id: uuid.UUID, _chunks: list[Document]) -> int:
        return 0

    def fake_loader(*_args: Any, **_kwargs: Any) -> list[Document]:
        raise DocumentProcessingError(
            "OCR engine unavailable",
            details={"error_code": CODE_OCR_ENGINE_UNAVAILABLE, "severity": "error"},
        )

    monkeypatch.setattr("src.api.ingestion_contract.load_document_from_upload", fake_loader)
    monkeypatch.setattr("src.api.ingestion_contract._store_documents", fake_store)

    result = await process_and_store_payloads(
        db=object(),  # type: ignore[arg-type]
        session_id=uuid.uuid4(),
        payloads=[_make_payload("photo.png", document_id="doc-img")],
        total_files=1,
        enable_ocr=True,
        extract_tables=False,
        logger=_DummyLogger(),
        initial_file_results=[],
    )

    assert result.ingestion_status == "failed"
    assert result.error_code == CODE_OCR_ENGINE_UNAVAILABLE
    assert result.file_results[0]["error_code"] == CODE_OCR_ENGINE_UNAVAILABLE
    assert result.file_results[0]["status"] == "failed"


def test_openapi_includes_structured_ingestion_fields() -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("asyncpg")
    from src.api.main import create_app

    app = create_app()
    schema = app.openapi()

    upload_props = schema["components"]["schemas"]["DocumentUploadResponse"]["properties"]
    session_props = schema["components"]["schemas"]["SessionInfo"]["properties"]

    assert "summary" in upload_props
    assert "file_results" in upload_props
    assert "ingestion_summary" in session_props
    assert "ingestion_warnings" in session_props
