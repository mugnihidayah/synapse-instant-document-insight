"""
Session management for RAG

Manages vectorstores per session to support multiple concurrent users
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Session


async def create_session(db: AsyncSession) -> Session:
  """Create a new session in database"""
  session = Session()
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
  stmt = select(Session).where(Session.expiry < now)
  result = await db.execute(stmt)
  expired = result.scalars().all()

  count = len(expired)
  for session in expired:
    await db.delete(session)

  return count