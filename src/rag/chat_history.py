"""
Chat history management for multiturn conversation
"""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logger import get_logger
from src.db.models import ChatMessage

logger = get_logger(__name__)


async def get_chat_history(
    db: AsyncSession,
    session_id: uuid.UUID,
    limit: int = 5,
) -> list[dict]:
    """
    get new chat history for sesssion

    Args:
        db: Database session
        session_id: Session ID
        limit: Number of message to retrieve

    Returns:
        List of chat message dict
    """
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(limit)
    )

    result = await db.execute(stmt)
    messages = result.scalars().all()

    # reverse for chronological order
    messages = list(reversed(messages))

    return [{"role": msg.role, "content": msg.content} for msg in messages]


async def save_chat_message(
    db: AsyncSession,
    session_id: uuid.UUID,
    role: str,
    content: str,
) -> ChatMessage:
    """
    Save message to chat history

    Args:
        db: Database session
        session_id: Session ID
        role: Role of message (user/assistant)
        content: Message content
    """
    message = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
    )
    db.add(message)
    await db.flush()

    logger.debug("chat_message_saved", session_id=str(session_id), role=role)

    return message


def format_chat_history(messages: list[dict]) -> str:
    """
    Format chat history into a string for prompt.

    Args:
        messages: List of chat message dict

    Returns:
        Formatted string
    """
    if not messages:
        return ""

    formatted = []
    for msg in messages:
        role = "Human" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content']}")

    return "\n".join(formatted)
