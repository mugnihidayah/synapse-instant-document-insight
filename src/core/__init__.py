"""
Core module containing shared utilities, configuration, and exceptions

This module provides:
- Configuration management using Pydantic settings
- Custom exception classes
- Shared constants and utilities
"""

from src.core.config import Settings, get_settings, settings
from src.core.exceptions import (
    ConfigurationError,
    DocumentProcessingError,
    RAGError,
    SynapseError,
    VectorStoreError,
)

__all__: list[str] = [
    "Settings",
    "get_settings",
    "settings",
    "SynapseError",
    "DocumentProcessingError",
    "VectorStoreError",
    "RAGError",
    "ConfigurationError",
]
