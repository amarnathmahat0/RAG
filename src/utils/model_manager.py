"""ModelManager — RAM-aware Ollama model lifecycle controller.

LATENCY FIXES
─────────────
FIX L1 — Warm model cache with TTL (biggest win)
    Previously keep_alive=0 cold-started every LLM call: load (~1.5s) +
    infer + unload. Three calls per query (router, query-transform, generator)
    = 4.5s+ of pure overhead before inference even starts.

    Now: keep_alive=-1 keeps model in Ollama RAM. A background loop evicts
    models after WARM_TTL_SEC=30s of inactivity. Second+ calls within TTL
    window pay ZERO load cost → router+transform+generate share one load.

FIX L2 — Skip redundant stop/start when model already running
    Old code stopped any running model before loading, even if it was the
    same model. Now we just refresh last_used_at and return immediately.

FIX L3 — /api/embed batched endpoint (Ollama >= 0.1.26)
    Kept from previous fix: POST /api/embed with input:[...] list.

FIX L4 — Harmless stop warnings demoted to DEBUG
    Kept from previous fix.
"""
from __future__ import annotations

import asyncio
import subprocess
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator

import httpx
import psutil

from src.utils.config import get_settings
from src.utils.exceptions import ModelLoadError, ModelUnloadError, RAMGuardrailError
from src.utils.logger import get_logger

logger = get_logger(__name__)
_settings = get_settings()

_LLM_MODELS = {"gemma3:1b", "qwen:1.8b", "phi3:mini"}
_LLM_KEYWORDS = ("gemma", "qwen", "phi", "llama", "mistral")

# Seconds to keep a model warm after last use before physically unloading.
# 30s covers router → query-transform → generator in one query session.
WARM_TTL_SEC = 30.0


@dataclass
class ModelRecord:
    name: str
    size_mb: float
    loaded_at: datetime = field(default_factory=datetime.utcnow)
    last_used_at: float = field(default_factory=time.monotonic)


class ModelManager:
    """Singleton-style manager; instantiate once and share."""

    def __init__(self) -> None:
        self._loaded: dict[str, ModelRecord] = {}
        self._lock = asyncio.Lock()
        self._client = httpx.AsyncClient(
            base_url=_settings.ollama_base_url,
            timeout=120.0,
        )
        self._evict_task: asyncio.Task | None = None
        self._ensure_eviction_loop()

    # ── FIX L1: TTL eviction loop ────────────────────────────────────────────────

    def _ensure_eviction_loop(self) -> None:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running() and self._evict_task is None:
                self._evict_task = loop.create_task(self._eviction_loop())
        except RuntimeError:
            pass

    async def _eviction_loop(self) -> None:
        while True:
            await asyncio.sleep(5)
            now = time.monotonic()
            to_evict: list[str] = []
            async with self._lock:
                for name, record in self._loaded.items():
                    if now - record.last_used_at > WARM_TTL_SEC:
                        to_evict.append(name)
                for name in to_evict:
                    logger.debug(
                        "TTL evict: model=%s idle=%.0fs", name,
                        now - self._loaded[name].last_used_at,
                    )
                    await self._unload_locked(name)

    # ── RAM helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def free_ram_mb() -> float:
        return psutil.virtual_memory().available / (1024 * 1024)

    @staticmethod
    def total_ram_mb() -> float:
        return psutil.virtual_memory().total / (1024 * 1024)

    def _check_ram(self, required_mb: float) -> None:
        free = self.free_ram_mb()
        logger.info(
            "RAM check | free=%.0fMB required=%.0fMB guardrail=%.0fMB",
            free, required_mb, _settings.min_free_ram_mb,
        )
        if free - required_mb < _settings.min_free_ram_mb:
            raise RAMGuardrailError(
                f"Insufficient RAM: {free:.0f}MB free, need {required_mb:.0f}MB + "
                f"{_settings.min_free_ram_mb:.0f}MB guardrail buffer."
            )

    async def _run_ollama_cmd(self, *args: str) -> str:
        proc = await asyncio.to_thread(
            subprocess.run,
            ("ollama",) + args,
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            raise ModelLoadError(
                f"ollama {' '.join(args)} failed: "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )
        return proc.stdout

    async def _list_running_models(self) -> set[str]:
        try:
            output = await self._run_ollama_cmd("ps")
            lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
            if len(lines) <= 1:
                return set()
            models: set[str] = set()
            for line in lines[1:]:
                parts = line.split()
                if parts:
                    name = parts[0]
                    models.add(name)
                    if name.endswith(":latest"):
                        models.add(name[: -len(":latest")])
            return models
        except Exception as exc:
            logger.warning("Could not list running Ollama models: %s", exc)
            return set()

    async def _stop_model_process(self, model: str) -> None:
        try:
            await self._run_ollama_cmd("stop", model)
        except ModelLoadError as exc:
            err_text = str(exc).lower()
            if "couldn't find" in err_text or "not found" in err_text:
                logger.debug("Model %s already stopped: %s", model, exc)
            else:
                logger.warning("Failed to stop model %s: %s", model, exc)
        except Exception as exc:
            logger.warning("Failed to stop model %s: %s", model, exc)

    # ── Load / unload ────────────────────────────────────────────────────────────

    async def load(self, model: str) -> None:
        """Load a model. FIX L1+L2: returns immediately if already warm."""
        # Ensure eviction loop is running (first call after event loop starts)
        self._ensure_eviction_loop()

        async with self._lock:
            # Already warm — refresh TTL, skip load entirely
            if model in self._loaded:
                self._loaded[model].last_used_at = time.monotonic()
                logger.debug("Model %s warm (TTL refreshed) — skipping load.", model)
                return

            running = await self._list_running_models()

            # Running in Ollama but not tracked — adopt it
            if model in running:
                size_mb = _settings.model_sizes_mb.get(model, 1000.0)
                self._loaded[model] = ModelRecord(name=model, size_mb=size_mb)
                logger.debug("Model %s adopted from running Ollama process.", model)
                return

            # Single-LLM policy: stop other LLMs before loading a new LLM
            model_is_llm = model in _LLM_MODELS or any(k in model for k in _LLM_KEYWORDS)
            if model_is_llm:
                for other in list(running):
                    other_is_llm = other in _LLM_MODELS or any(
                        k in other for k in _LLM_KEYWORDS
                    )
                    if other_is_llm and other != model:
                        logger.info("Evicting LLM %s before loading %s", other, model)
                        await self._stop_model_process(other)
                        self._loaded.pop(other, None)

            size_mb = _settings.model_sizes_mb.get(model, 1000.0)
            self._check_ram(size_mb)

            t0 = time.perf_counter()
            try:
                await self._client.post(
                    "/api/generate",
                    json={
                        "model": model,
                        "prompt": "",
                        "keep_alive": -1,   # FIX L1: keep in RAM
                        "stream": False,
                    },
                    timeout=120.0,
                )
            except Exception as exc:
                raise ModelLoadError(f"Failed to load {model}: {exc}") from exc

            elapsed = time.perf_counter() - t0
            self._loaded[model] = ModelRecord(name=model, size_mb=size_mb)
            logger.info(
                "LOAD model=%s size_mb=%.0f elapsed=%.2fs free_ram=%.0fMB ts=%s",
                model, size_mb, elapsed, self.free_ram_mb(),
                datetime.utcnow().isoformat(),
            )

    async def unload(self, model: str) -> None:
        """FIX L1: Deferred unload — just touch last_used_at, TTL loop evicts later."""
        async with self._lock:
            if model in self._loaded:
                self._loaded[model].last_used_at = time.monotonic()
                logger.debug("unload(%s) deferred — TTL=%.0fs", model, WARM_TTL_SEC)

    async def _unload_locked(self, model: str) -> None:
        """Physical unload. Must be called while _lock is held."""
        running = await self._list_running_models()
        if model not in running and model not in self._loaded:
            return
        try:
            await self._stop_model_process(model)
        except Exception as exc:
            raise ModelUnloadError(f"Failed to unload {model}: {exc}") from exc
        finally:
            record = self._loaded.pop(model, None)
            if record:
                logger.info(
                    "UNLOAD model=%s was_loaded_at=%s free_ram=%.0fMB ts=%s",
                    model, record.loaded_at.isoformat(),
                    self.free_ram_mb(), datetime.utcnow().isoformat(),
                )

    @asynccontextmanager
    async def use_model(self, model: str) -> AsyncIterator[str]:
        """Load → yield → deferred unload."""
        await self.load(model)
        try:
            yield model
        finally:
            await self.unload(model)

    # ── Inference helpers ────────────────────────────────────────────────────────

    async def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        async with self.use_model(model):
            payload: dict = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
                "keep_alive": -1,
            }
            if system:
                payload["system"] = system
            try:
                resp = await self._client.post(
                    "/api/generate", json=payload, timeout=120.0
                )
                resp.raise_for_status()
                return resp.json().get("response", "")
            except Exception as exc:
                raise ModelLoadError(f"Generation failed for {model}: {exc}") from exc

    async def generate_stream(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        import json
        await self.load(model)
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "keep_alive": -1,
        }
        if system:
            payload["system"] = system
        try:
            async with self._client.stream(
                "POST", "/api/generate", json=payload, timeout=180.0
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.strip():
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done"):
                            break
        finally:
            await self.unload(model)

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        """FIX L3: /api/embed batched endpoint. Falls back to /api/embeddings."""
        if not texts:
            return []

        await self.load(model)
        try:
            # Batched /api/embed (Ollama >= 0.1.26)
            try:
                resp = await self._client.post(
                    "/api/embed",
                    json={"model": model, "input": texts},
                    timeout=60.0,
                )
                resp.raise_for_status()
                data = resp.json()
                embeddings = data.get("embeddings")
                if embeddings and len(embeddings) == len(texts):
                    return embeddings
                logger.warning(
                    "Batched /api/embed unexpected shape %s; falling back",
                    list(data.keys()),
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 400:
                    try:
                        body = exc.response.json()
                        logger.error("Ollama error 400: %s", body)
                    except Exception:
                        pass
                    raise
                if exc.response.status_code != 404:
                    raise
                logger.debug("/api/embed 404; trying legacy /api/embeddings")

            # Legacy per-text /api/embeddings
            embeddings = []
            for text in texts:
                resp = await self._client.post(
                    "/api/embeddings",
                    json={"model": model, "prompt": text},
                    timeout=60.0,
                )
                resp.raise_for_status()
                embeddings.append(resp.json()["embedding"])
            return embeddings

        finally:
            await self.unload(model)

    async def status(self) -> dict:
        return {
            "loaded_models": [
                {"name": r.name, "size_mb": r.size_mb, "loaded_at": r.loaded_at.isoformat()}
                for r in self._loaded.values()
            ],
            "free_ram_mb": round(self.free_ram_mb(), 1),
            "total_ram_mb": round(self.total_ram_mb(), 1),
            "guardrail_mb": _settings.min_free_ram_mb,
        }

    async def close(self) -> None:
        if self._evict_task:
            self._evict_task.cancel()
        async with self._lock:
            for name in list(self._loaded.keys()):
                await self._unload_locked(name)
        await self._client.aclose()


# ── Module-level singleton ───────────────────────────────────────────────────────
_manager: ModelManager | None = None


def get_model_manager() -> ModelManager:
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager


OllamaProcessManager = ModelManager