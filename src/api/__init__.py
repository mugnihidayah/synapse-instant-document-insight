"""
API module for FastAPI endpoints

This module provides:
- REST API for document management
- RAG query endpoint
- Health check
"""

from typing import Any


def create_app() -> Any:
    """Create FastAPI app lazily to avoid hard import at package import time."""
    from src.api.main import create_app as _create_app

    return _create_app()


try:
    from src.api.main import app
except ModuleNotFoundError:  # pragma: no cover - optional API dependency for core tests
    app = None

__all__ = [
    "app",
    "create_app",
]
