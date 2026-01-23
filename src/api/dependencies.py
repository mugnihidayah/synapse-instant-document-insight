"""
FastAPI dependencies for authentication
"""

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import validate_api_key
from src.db import get_db
from src.db.models import APIKey

# header name for API key
API_KEY_HEADER = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


async def get_api_key(
    api_key: str | None = Security(API_KEY_HEADER),
    db: AsyncSession = Depends(get_db),
) -> APIKey:
    """
    Dependency to validate API key from header

    Usage:
      @app.get("/protected")
      async def protected_route(api_key: APIKey = Depends(get_api_key)):
          ...
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Include X-API key header",
        )

    valid_key = await validate_api_key(db, api_key)

    if not valid_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoke API key",
        )

    return valid_key


async def get_optional_api_key(
    api_key: str | None = Security(API_KEY_HEADER),
    db: AsyncSession = Depends(get_db),
) -> APIKey | None:
    """Optional API key"""
    if not api_key:
        return None

    return await validate_api_key(db, api_key)
