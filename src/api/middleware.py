"""FastAPI middleware: CORS, rate limiting, request ID, timing headers."""
from __future__ import annotations

import time
import uuid
from collections import defaultdict
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from src.monitoring.prometheus_metrics import update_system_gauges
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
_settings = get_settings()


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:16]
        request.state.request_id = req_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-process sliding window rate limiter (no Redis required)."""

    def __init__(self, app, requests_per_window: int = 10, window_sec: int = 60) -> None:
        super().__init__(app)
        self._limit = requests_per_window
        self._window = window_sec
        self._buckets: dict[str, list[float]] = defaultdict(list)

    def _get_ip(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Only rate-limit the /query endpoint
        if not request.url.path.startswith("/query"):
            return await call_next(request)

        ip = self._get_ip(request)
        now = time.time()
        window_start = now - self._window

        # Prune old entries
        self._buckets[ip] = [t for t in self._buckets[ip] if t > window_start]

        if len(self._buckets[ip]) >= self._limit:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Max {self._limit} requests per {self._window}s",
                },
                headers={"Retry-After": str(self._window)},
            )

        self._buckets[ip].append(now)
        return await call_next(request)


def register_middleware(app: FastAPI) -> None:
    """Register all middleware in the correct order."""
    # CORS (outermost)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Rate limiter
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_window=_settings.rate_limit_requests,
        window_sec=_settings.rate_limit_window,
    )
    # Timing
    app.add_middleware(TimingMiddleware)
    # Request ID (innermost — runs last in response, first in request)
    app.add_middleware(RequestIDMiddleware)
