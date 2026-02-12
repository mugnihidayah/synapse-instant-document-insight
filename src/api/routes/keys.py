"""
API key management endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import create_api_key, revoke_api_key
from src.api.dependencies import get_api_key
from src.db import get_db
from src.db.models import APIKey

router = APIRouter(prefix="/keys", tags=["API Keys"])


class CreateKeyRequest(BaseModel):
    name: str | None = None
    rate_limit: int = 100


class CreateKeyResponse(BaseModel):
    api_key: str
    key_id: str
    name: str | None
    rate_limit: int
    message: str = "Store this key securely. It will not be shown again."


class KeyInfo(BaseModel):
    key_id: str
    name: str | None
    rate_limit: int
    is_active: bool
    created_at: str
    last_used_at: str | None


@router.post("/", response_model=CreateKeyResponse)
async def create_key(
    request: CreateKeyRequest,
    db: AsyncSession = Depends(get_db),
) -> CreateKeyResponse:
    """
    Create a new API key

    WARNING: The API key is only show once. Store it securely.
    """
    plain_key, api_key = await create_api_key(
        db,
        name=request.name,
        rate_limit=request.rate_limit,
    )

    return CreateKeyResponse(
        api_key=plain_key,
        key_id=str(api_key.id),
        name=api_key.name,
        rate_limit=api_key.rate_limit,
    )


@router.get("/", response_model=list[KeyInfo])
async def get_keys(
    db: AsyncSession = Depends(get_db),
    api_key: APIKey = Depends(get_api_key),
) -> list[KeyInfo]:
    """List all API keys"""

    return [
        KeyInfo(
            key_id=str(api_key.id),
            name=api_key.name,
            rate_limit=api_key.rate_limit,
            is_active=api_key.is_active,
            created_at=api_key.created_at.isoformat(),
            last_used_at=api_key.last_used_at.isoformat() if api_key.last_used_at else None,
        )
    ]


@router.delete("/{key_id}")
async def delete_key(
    key_id: str,
    db: AsyncSession = Depends(get_db),
    api_key: APIKey = Depends(get_api_key),
) -> dict:
    """Revoke an API key"""
    import uuid

    try:
        uid = uuid.UUID(key_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid key ID format",
        ) from None

    if uid != api_key.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to revoke this API key",
        )

    revoked = await revoke_api_key(db, uid)

    if not revoked:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API Key not found",
        )

    return {"message": f"API key {key_id} revoked"}
