"""Feedback, usage, and export endpoints."""

import json

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import PlainTextResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api import session as session_service
from src.api.dependencies import get_api_key
from src.api.schemas import FeedbackRequest, FeedbackResponse, UsageResponse
from src.api.usage import get_usage_summary, record_usage_event
from src.core.config import settings
from src.db import get_db
from src.db.models import APIKey, ChatMessage, Feedback

router = APIRouter(prefix="/insights", tags=["Insights"])


@router.post("/feedback/{session_id}", response_model=FeedbackResponse)
async def submit_feedback(
    session_id: str,
    payload: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
    api_key: APIKey = Depends(get_api_key),
) -> FeedbackResponse:
    """Submit answer quality feedback for a session."""
    session = await session_service.get_session_by_str(db, session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    if session.api_key_id != api_key.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this session",
        )

    feedback = Feedback(
        session_id=session.id,
        api_key_id=api_key.id,
        question=payload.question,
        answer=payload.answer,
        rating=payload.rating,
        comment=payload.comment,
        metadata_=payload.metadata,
    )
    db.add(feedback)
    await db.flush()

    await record_usage_event(
        db,
        api_key.id,
        "feedback",
        session_id=session.id,
        metadata={"rating": payload.rating},
    )

    return FeedbackResponse(
        feedback_id=str(feedback.id),
        session_id=session_id,
        rating=feedback.rating,
        created_at=feedback.created_at,
    )


@router.get("/usage", response_model=UsageResponse)
async def usage_summary(
    db: AsyncSession = Depends(get_db),
    api_key: APIKey = Depends(get_api_key),
) -> UsageResponse:
    """Get usage analytics and daily quota summary."""
    summary = await get_usage_summary(db, api_key.id)
    return UsageResponse(key_id=str(api_key.id), **summary)


@router.get("/export/{session_id}")
async def export_session(
    session_id: str,
    format: str = Query(default="markdown", pattern="^(markdown|json)$"),
    db: AsyncSession = Depends(get_db),
    api_key: APIKey = Depends(get_api_key),
) -> Response:
    """Export chat history for a session as markdown or JSON."""
    session = await session_service.get_session_by_str(db, session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    if session.api_key_id != api_key.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this session",
        )

    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.created_at.asc())
        .limit(settings.export_max_messages)
    )
    messages = list((await db.execute(stmt)).scalars().all())

    payload = {
        "session_id": session_id,
        "created_at": session.created_at.isoformat(),
        "message_count": len(messages),
        "messages": [
            {
                "role": item.role,
                "content": item.content,
                "created_at": item.created_at.isoformat(),
            }
            for item in messages
        ],
    }

    await record_usage_event(
        db,
        api_key.id,
        "export",
        session_id=session.id,
        metadata={"format": format, "messages": len(messages)},
    )

    if format == "json":
        return Response(
            content=json.dumps(payload, ensure_ascii=False, indent=2),
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename=session-{session_id}.json",
            },
        )

    lines = [
        f"# Session Export {session_id}",
        "",
        f"Created at: {session.created_at.isoformat()}",
        f"Messages: {len(messages)}",
        "",
    ]
    for message in messages:
        role = "User" if message.role == "user" else "Assistant"
        lines.append(f"## {role} ({message.created_at.isoformat()})")
        lines.append("")
        lines.append(message.content)
        lines.append("")

    markdown_text = "\n".join(lines)
    return PlainTextResponse(
        content=markdown_text,
        headers={
            "Content-Disposition": f"attachment; filename=session-{session_id}.md",
        },
        media_type="text/markdown",
    )
