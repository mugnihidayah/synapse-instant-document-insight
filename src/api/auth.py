"""
API key authentication
"""

import hashlib
import secrets
import uuid
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import APIKey


def generate_api_key():
  """Generate a new API key"""
  return f"sk-{secrets.token_urlsafe(32)}"

def hash_api_key(key: str) -> str:
  """Hash an API key"""
  return hashlib.sha256(key.encode()).hexdigest()

async def create_api_key(
  db: AsyncSession,
  name: str | None = None,
  rate_limit: int = 100,
) -> tuple[str, APIKey]:
  """
  Create a new API key

  Returns:
    tuple of (plain_key, api_key_record)

  IMPORTANT! plain_key is only returned once. Store it securely.
  """
  plain_key = generate_api_key()
  key_hash = hash_api_key(plain_key)

  api_key = APIKey(
    key_hash=key_hash,
    name=name,
    rate_limit=rate_limit,
  )

  db.add(api_key)
  await db.flush()

  return plain_key, api_key

async def validate_api_key(
  db: AsyncSession,
  key: str,
) -> APIKey | None:
  """
  Validate an API key

  Returns:
    APIKey record if valid, None otherwise
  """

  key_hash = hash_api_key(key)

  stmt = select(APIKey).where(
    APIKey.key_hash == key_hash,
    APIKey.is_active,
  )

  result = await db.execute(stmt)
  api_key = result.scalar_one_or_none()

  if api_key:
    # update last used timestamp
    await db.execute(
      update(APIKey)
      .where(APIKey.id == api_key.id)
      .values(last_used_at=datetime.now(UTC))
    )

  return api_key

async def revoke_api_key(
  db: AsyncSession,
  key_id: uuid.UUID,
) -> bool:
  """Revoke an API key"""
  api_key = await db.get(APIKey, key_id)

  if api_key:
    api_key.is_active = False
    return True

  return False

async def list_api_keys(db: AsyncSession) -> list[APIKey]:
  """List all API keys"""
  stmt = select(APIKey).order_by(APIKey.created_at.desc())
  result = await db.execute(stmt)
  return list(result.scalars().all())