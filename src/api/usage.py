"""Usage tracking helpers for analytics and quota enforcement."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.db.models import Feedback, Session, UsageEvent


async def record_usage_event(
    db: AsyncSession,
    api_key_id: uuid.UUID,
    event_type: str,
    *,
    session_id: uuid.UUID | None = None,
    metadata: dict | None = None,
) -> UsageEvent:
    event = UsageEvent(
        api_key_id=api_key_id,
        session_id=session_id,
        event_type=event_type,
        metadata_=metadata or {},
    )
    db.add(event)
    await db.flush()
    return event


async def count_events_today(
    db: AsyncSession,
    api_key_id: uuid.UUID,
    event_type: str,
) -> int:
    today = datetime.now(UTC).date()

    stmt = select(func.count(UsageEvent.id)).where(
        UsageEvent.api_key_id == api_key_id,
        UsageEvent.event_type == event_type,
        func.date(UsageEvent.created_at) == today,
    )
    result = await db.execute(stmt)
    count = result.scalar_one()
    return int(count or 0)


async def get_usage_summary(db: AsyncSession, api_key_id: uuid.UUID) -> dict:
    sessions_stmt = select(func.count(Session.id)).where(Session.api_key_id == api_key_id)
    sessions_result = await db.execute(sessions_stmt)
    total_sessions = int(sessions_result.scalar_one() or 0)

    docs_stmt = select(func.coalesce(func.sum(Session.document_count), 0)).where(
        Session.api_key_id == api_key_id
    )
    docs_result = await db.execute(docs_stmt)
    total_documents = int(docs_result.scalar_one() or 0)

    queries_stmt = select(func.count(UsageEvent.id)).where(
        UsageEvent.api_key_id == api_key_id,
        UsageEvent.event_type == "query",
    )
    queries_result = await db.execute(queries_stmt)
    total_queries = int(queries_result.scalar_one() or 0)

    feedback_stmt = select(func.count(Feedback.id)).where(Feedback.api_key_id == api_key_id)
    feedback_result = await db.execute(feedback_stmt)
    total_feedback = int(feedback_result.scalar_one() or 0)

    used_today = await count_events_today(db, api_key_id, "query")
    remaining_today = max(0, settings.usage_daily_query_quota - used_today)

    return {
        "total_sessions": total_sessions,
        "total_queries": total_queries,
        "total_documents": total_documents,
        "total_feedback": total_feedback,
        "quota": {
            "daily_limit": settings.usage_daily_query_quota,
            "used_today": used_today,
            "remaining_today": remaining_today,
        },
    }


async def has_remaining_query_quota(db: AsyncSession, api_key_id: uuid.UUID) -> bool:
    used_today = await count_events_today(db, api_key_id, "query")
    return used_today < settings.usage_daily_query_quota
