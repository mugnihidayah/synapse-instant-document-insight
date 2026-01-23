"""
Database module for PostgreSQL + pgvector
Provides:
- Async SQLAlchemy connection
- ORM models for sessions, api_keys, documents
- Vector similarity search via pgvector
"""
from src.db.connection import async_session_maker, engine, get_db, get_db_context
from src.db.models import APIKey, Base, Document, Session

__all__ = [
    "engine",
    "async_session_maker",
    "get_db",
    "get_db_context",
    "Base",
    "Session",
    "APIKey",
    "Document",
]