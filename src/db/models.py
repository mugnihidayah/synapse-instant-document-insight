"""
SQLAlchemy models for PostgreSQL + pgvector
"""
import uuid
from datetime import datetime, timedelta, timezone
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
  Boolean,
  DateTime,
  ForeignKey,
  Integer,
  String,
  Text,
  func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
class Base(DeclarativeBase):
  """Base class for all models"""
  pass
class Session(Base):
  """User session for document storage"""
    
  __tablename__ = "sessions"
    
  id: Mapped[uuid.UUID] = mapped_column(
    UUID(as_uuid=True),
    primary_key=True,
    default=uuid.uuid4,
  )
  created_at: Mapped[datetime] = mapped_column(
    DateTime(timezone=True),
    server_default=func.now(),
  )
  expires_at: Mapped[datetime] = mapped_column(
    DateTime(timezone=True),
    default=lambda: datetime.now(timezone.utc) + timedelta(hours=24),
  )
  document_count: Mapped[int] = mapped_column(
    Integer,
    default=0,
  )
  metadata_: Mapped[dict] = mapped_column(
    "metadata",
    JSONB,
    default=dict,
  )
    
  # Relationship - MUST be after Document class is defined
  documents: Mapped[list["Document"]] = relationship(
    back_populates="session",
    cascade="all, delete-orphan",
  )
class APIKey(Base):
  """API key for authentication"""
    
  __tablename__ = "api_keys"
    
  id: Mapped[uuid.UUID] = mapped_column(
    UUID(as_uuid=True),
    primary_key=True,
    default=uuid.uuid4,
  )
  key_hash: Mapped[str] = mapped_column(
    String(64),
    unique=True,
    nullable=False,
  )
  name: Mapped[str | None] = mapped_column(
    String(100),
    nullable=True,
  )
  rate_limit: Mapped[int] = mapped_column(
    Integer,
    default=100,
  )
  is_active: Mapped[bool] = mapped_column(
    Boolean,
    default=True,
  )
  created_at: Mapped[datetime] = mapped_column(
    DateTime(timezone=True),
    server_default=func.now(),
  )
  last_used_at: Mapped[datetime | None] = mapped_column(
    DateTime(timezone=True),
    nullable=True,
  )
class Document(Base):
  """Document with vector embedding"""
    
  __tablename__ = "documents"
    
  id: Mapped[uuid.UUID] = mapped_column(
    UUID(as_uuid=True),
    primary_key=True,
    default=uuid.uuid4,
  )
  session_id: Mapped[uuid.UUID] = mapped_column(
    UUID(as_uuid=True),
    ForeignKey("sessions.id", ondelete="CASCADE"),
    nullable=False,
  )
  content: Mapped[str] = mapped_column(
    Text,
    nullable=False,
  )
  embedding: Mapped[list[float] | None] = mapped_column(
    Vector(384),
    nullable=True,
  )
  metadata_: Mapped[dict] = mapped_column(
    "metadata",
    JSONB,
    default=dict,
  )
  created_at: Mapped[datetime] = mapped_column(
    DateTime(timezone=True),
    server_default=func.now(),
  )
    
  # Relationship back to Session
  session: Mapped["Session"] = relationship(
    back_populates="documents",
  )