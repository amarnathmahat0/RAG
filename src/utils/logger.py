"""Structured JSON logger for NexusRAG."""
from __future__ import annotations

import logging
import sys
from typing import Any

try:
    import structlog

    _USE_STRUCTLOG = True
except ImportError:
    _USE_STRUCTLOG = False


def _setup_stdlib_logging(level: str = "INFO") -> None:
    fmt = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        stream=sys.stdout,
    )


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a stdlib logger (structlog optional)."""
    _setup_stdlib_logging(level)
    return logging.getLogger(name)


class PipelineLogger:
    """Thin wrapper that adds pipeline context to every log call."""

    def __init__(self, name: str) -> None:
        self._log = get_logger(name)
        self._context: dict[str, Any] = {}

    def bind(self, **kwargs: Any) -> "PipelineLogger":
        self._context.update(kwargs)
        return self

    def _fmt(self, msg: str) -> str:
        ctx = " | ".join(f"{k}={v}" for k, v in self._context.items())
        return f"{msg} [{ctx}]" if ctx else msg

    def info(self, msg: str, **kw: Any) -> None:
        self._log.info(self._fmt(msg), extra=kw)

    def debug(self, msg: str, **kw: Any) -> None:
        self._log.debug(self._fmt(msg), extra=kw)

    def warning(self, msg: str, **kw: Any) -> None:
        self._log.warning(self._fmt(msg), extra=kw)

    def error(self, msg: str, **kw: Any) -> None:
        self._log.error(self._fmt(msg), extra=kw)

    def exception(self, msg: str, **kw: Any) -> None:
        self._log.exception(self._fmt(msg), extra=kw)
