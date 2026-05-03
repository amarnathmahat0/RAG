"""Centralized logging configuration."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_logging(level: str = "INFO", log_file: str | None = None) -> None:
    fmt = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=handlers,
        force=True,
    )
    # Silence noisy third-party loggers
    for noisy in ["httpx", "httpcore", "chromadb", "neo4j"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
