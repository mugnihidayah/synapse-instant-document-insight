"""Tests for uploaded file storage utilities."""

import shutil
import uuid
from pathlib import Path

import pytest

from src.core.config import settings
from src.ingestion.file_storage import resolve_original_file_path, save_original_file


def _make_workspace_temp_dir() -> Path:
    root = Path("tests") / ".tmp_storage"
    root.mkdir(parents=True, exist_ok=True)
    path = root / str(uuid.uuid4())
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_save_original_file_persists_content() -> None:
    tmp_path = _make_workspace_temp_dir()
    original_upload_dir = settings.upload_dir
    settings.upload_dir = tmp_path

    try:
        content = b"sample data"
        info = save_original_file(
            session_id="session-1",
            filename="report.pdf",
            content=content,
            document_id="doc-1",
        )

        stored_path = tmp_path / info.file_path
        assert info.document_id == "doc-1"
        assert info.original_filename == "report.pdf"
        assert info.file_size_bytes == len(content)
        assert stored_path.exists()
        assert stored_path.read_bytes() == content
    finally:
        settings.upload_dir = original_upload_dir
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_resolve_original_file_path_rejects_traversal() -> None:
    tmp_path = _make_workspace_temp_dir()
    original_upload_dir = settings.upload_dir
    settings.upload_dir = tmp_path

    try:
        with pytest.raises(ValueError):
            resolve_original_file_path("../outside.txt")
    finally:
        settings.upload_dir = original_upload_dir
        shutil.rmtree(tmp_path, ignore_errors=True)
