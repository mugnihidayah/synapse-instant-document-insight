"""
Core module containing shared utilities, configuration, and exceptions

This module provides:
- Configuration management using Pydantic settings
- Custom exception classes
- Shared constants and utilities
"""

from src.core.config import settings, Settings, get_settings
from src.core.exceptions import (
  SynapseError,
  DocumentProcessingError,
  VectorStoreError,
  RAGError,
  ConfigurationError,
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