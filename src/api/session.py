"""
Session management for RAG

Manages vectorstores per session to support multiple concurrent users
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.ingestion_contract import (
    FileIngestionResultData,
    IngestionSummaryData,
    IngestionWarningData,
    empty_ingestion_summary,
    normalize_file_results,
    normalize_ingestion_summary,
    normalize_ingestion_warnings,
)
from src.db.models import Session


async def create_session(db: AsyncSession, api_key_id: uuid.UUID) -> Session:
    """Create a new session in database"""
    session = Session(api_key_id=api_key_id)
    db.add(session)
    await db.flush()
    return session


async def get_session(db: AsyncSession, session_id: uuid.UUID) -> Session | None:
    """Get session by ID"""
    return await db.get(Session, session_id)


async def get_session_by_str(db: AsyncSession, session_id: str) -> Session | None:
    """Get session by string ID"""
    try:
        uid = uuid.UUID(session_id)
        return await get_session(db, uid)
    except ValueError:
        return None


async def delete_session(db: AsyncSession, session_id: uuid.UUID) -> bool:
    """Delete a session"""
    session = await get_session(db, session_id)
    if session:
        await db.delete(session)
        return True
    return False


async def cleanup_expired_sessions(db: AsyncSession) -> int:
    """Delete all expired sessions"""
    now = datetime.now(UTC)
    stmt = select(Session).where(Session.expires_at < now)
    result = await db.execute(stmt)
    expired = result.scalars().all()

    count = len(expired)
    for session in expired:
        await db.delete(session)

    return count


async def get_session_for_key(
    db: AsyncSession, session_id: uuid.UUID, api_key_id: uuid.UUID
) -> Session | None:
    """Get session for a specific API key"""
    session = await get_session(db, session_id)
    if session and session.api_key_id == api_key_id:
        return session
    return None


async def set_ingestion_status(
    db: AsyncSession,
    session_id: uuid.UUID,
    status: str,
    *,
    error: str | None = None,
    summary: IngestionSummaryData | None = None,
    warnings: list[IngestionWarningData] | None = None,
    file_results: list[FileIngestionResultData] | None = None,
    error_code: str | None = None,
) -> Session | None:
    """Update ingestion status and timestamps for a session."""
    session = await get_session(db, session_id)
    if not session:
        return None

    now = datetime.now(UTC)
    session.ingestion_status = status
    session.ingestion_error = error

    metadata = dict(session.metadata_ or {})
    if summary is not None:
        metadata["ingestion_summary"] = normalize_ingestion_summary(summary)
    if warnings is not None:
        metadata["ingestion_warnings"] = normalize_ingestion_warnings(warnings)
    if file_results is not None:
        metadata["ingestion_file_results"] = normalize_file_results(file_results)
    if error_code is not None or status in {"ready", "ready_with_warnings", "queued", "processing"}:
        metadata["ingestion_error_code"] = error_code
    session.metadata_ = metadata

    if status == "processing":
        session.ingestion_started_at = now
        session.ingestion_completed_at = None
    elif status in {"ready", "ready_with_warnings", "failed"}:
        if session.ingestion_started_at is None:
            session.ingestion_started_at = now
        session.ingestion_completed_at = now

    await db.flush()
    return session


def get_ingestion_summary(session: Session) -> IngestionSummaryData:
    """Get normalized ingestion summary from session metadata."""
    metadata = session.metadata_ or {}
    return (
        normalize_ingestion_summary(metadata.get("ingestion_summary")) or empty_ingestion_summary()
    )


def get_ingestion_warnings(session: Session) -> list[IngestionWarningData]:
    """Get normalized ingestion warnings from session metadata."""
    metadata = session.metadata_ or {}
    return normalize_ingestion_warnings(metadata.get("ingestion_warnings"))


def get_ingestion_file_results(session: Session) -> list[FileIngestionResultData]:
    """Get normalized per-file ingestion outcomes from session metadata."""
    metadata = session.metadata_ or {}
    return normalize_file_results(metadata.get("ingestion_file_results"))


def get_ingestion_error_code(session: Session) -> str | None:
    """Get session-level ingestion error code from session metadata."""
    metadata = session.metadata_ or {}
    value = metadata.get("ingestion_error_code")
    return str(value) if value else None
