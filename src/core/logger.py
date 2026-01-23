"""
Structured logging using struclog
"""

import logging
import sys

import structlog

from src.core.config import settings


def setup_logging():
    """Configure structured logging"""

    # Set log level
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )


def get_logger(name: str = __name__) -> structlog.BoundLogger:
    """Get a logger instance"""
    return structlog.get_logger(name)
