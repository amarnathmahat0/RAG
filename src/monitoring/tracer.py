"""File-based JSONL request tracer — no external dependencies."""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Generator

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
_settings = get_settings()


@dataclass
class Span:
    name: str
    start_ms: float = field(default_factory=lambda: time.perf_counter() * 1000)
    end_ms: float = 0.0
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def finish(self, **metadata: Any) -> None:
        self.end_ms = time.perf_counter() * 1000
        self.duration_ms = round(self.end_ms - self.start_ms, 2)
        self.metadata.update(metadata)

    def add_event(self, name: str, **data: Any) -> None:
        self.events.append({
            "name": name,
            "ts_ms": round(time.perf_counter() * 1000, 2),
            **data,
        })


@dataclass
class Trace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    query: str = ""
    start_ms: float = field(default_factory=lambda: time.perf_counter() * 1000)
    end_ms: float = 0.0
    total_duration_ms: float = 0.0
    spans: list[Span] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"

    def add_span(self, name: str) -> Span:
        span = Span(name=name)
        self.spans.append(span)
        return span

    def finish(self, status: str = "ok", **metadata: Any) -> None:
        self.end_ms = time.perf_counter() * 1000
        self.total_duration_ms = round(self.end_ms - self.start_ms, 2)
        self.status = status
        self.metadata.update(metadata)


class Tracer:
    """Write one JSONL line per request to the trace file."""

    def __init__(self) -> None:
        self._trace_dir = Path(_settings.trace_dir)
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self._trace_dir / "traces.jsonl"

    def new_trace(self, query: str, request_id: str | None = None) -> Trace:
        t = Trace(query=query)
        if request_id:
            t.trace_id = request_id
        return t

    def save(self, trace: Trace) -> None:
        try:
            record = asdict(trace)
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:
            logger.warning("Failed to save trace: %s", exc)

    @contextmanager
    def span(self, trace: Trace, name: str, **meta: Any) -> Generator[Span, None, None]:
        s = trace.add_span(name)
        s.metadata.update(meta)
        try:
            yield s
        except Exception as exc:
            s.error = str(exc)
            s.finish()
            raise
        else:
            s.finish()

    def read_traces(self, last_n: int = 50) -> list[dict]:
        """Read last N traces from JSONL file."""
        if not self._log_file.exists():
            return []
        lines = self._log_file.read_text().strip().split("\n")
        lines = [l for l in lines if l.strip()]
        return [json.loads(l) for l in lines[-last_n:]]


_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer
