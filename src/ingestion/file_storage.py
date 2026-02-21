"""Utilities for durable storage of original uploaded files."""

from __future__ import annotations

import mimetypes
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from src.core.config import settings

_FILENAME_SANITIZER = re.compile(r"[^a-zA-Z0-9._-]")


@dataclass(frozen=True)
class StoredFileInfo:
    """Metadata describing a persisted uploaded file."""

    document_id: str
    original_filename: str
    file_path: str
    mime_type: str
    file_size_bytes: int


def _safe_filename(filename: str) -> str:
    clean = Path(filename).name.strip()
    if not clean:
        return "file.bin"

    sanitized = _FILENAME_SANITIZER.sub("_", clean)
    return sanitized[:255] or "file.bin"


def _resolve_candidate_path(file_path: str) -> Path:
    candidate = Path(file_path)
    if not candidate.is_absolute():
        candidate = settings.upload_dir / candidate

    resolved_candidate = candidate.resolve()
    resolved_root = settings.upload_dir.resolve()

    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("file_path is outside configured upload_dir") from exc

    return resolved_candidate


def save_original_file(
    *,
    session_id: str,
    filename: str,
    content: bytes,
    document_id: str | None = None,
) -> StoredFileInfo:
    """Persist uploaded file and return storage metadata."""
    doc_id = document_id or str(uuid.uuid4())
    display_name = Path(filename).name.strip() or "file.bin"
    safe_name = _safe_filename(display_name)

    relative_path = Path(session_id) / doc_id / safe_name
    absolute_path = (settings.upload_dir / relative_path).resolve()
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_path.write_bytes(content)

    guessed_mime, _ = mimetypes.guess_type(safe_name)
    mime_type = guessed_mime or "application/octet-stream"

    return StoredFileInfo(
        document_id=doc_id,
        original_filename=display_name,
        file_path=relative_path.as_posix(),
        mime_type=mime_type,
        file_size_bytes=len(content),
    )


def resolve_original_file_path(file_path: str) -> Path:
    """Resolve stored file path and enforce upload_dir sandbox."""
    return _resolve_candidate_path(file_path)
