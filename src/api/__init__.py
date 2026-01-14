"""
API module for FastAPI endpoints

This module provides:
- REST API for document management
- RAG query endpoint
- Health check
"""

from src.api.main import app, create_app

__all__ = [
    "app",
    "create_app",
]
