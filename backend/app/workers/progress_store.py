"""Ephemeral key/value store for live job progress and cancel flags.

Two implementations, selected by `settings.ALF_TASK_QUEUE`:

  - `local`  → `InMemoryStore`. The desktop app's FastAPI process *is* the
    worker, so an in-process dict is not merely adequate — it is strictly
    correct: the job thread that writes progress and the SSE handler that
    reads it share the same memory. No Redis.
  - `celery` → `RedisStore`. The server deployment runs the API and the
    Celery worker as separate processes, so progress has to cross a
    process boundary.

`workers/progress.py` and `workers/training_progress.py` are the only
callers; they keep their existing public API and just delegate here.
"""
from __future__ import annotations

import threading
import time
from typing import Protocol

from app.core.config import settings

_DEFAULT_TTL_S = 3600


class ProgressStore(Protocol):
    def set(self, key: str, value: str, ttl_s: int = _DEFAULT_TTL_S) -> None: ...
    def get(self, key: str) -> str | None: ...
    def delete(self, key: str) -> None: ...
    def exists(self, key: str) -> bool: ...


class InMemoryStore:
    """Process-local, thread-safe, with lazy TTL expiry."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()

    def _sweep_locked(self, now: float) -> None:
        for k in [k for k, (_, exp) in self._data.items() if exp <= now]:
            self._data.pop(k, None)

    def set(self, key: str, value: str, ttl_s: int = _DEFAULT_TTL_S) -> None:
        now = time.monotonic()
        with self._lock:
            self._sweep_locked(now)
            self._data[key] = (value, now + ttl_s)

    def get(self, key: str) -> str | None:
        now = time.monotonic()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            value, exp = item
            if exp <= now:
                self._data.pop(key, None)
                return None
            return value

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def exists(self, key: str) -> bool:
        return self.get(key) is not None


class RedisStore:
    def __init__(self, url: str) -> None:
        import redis  # server-only dependency

        self._redis = redis.from_url(url, decode_responses=True)

    def set(self, key: str, value: str, ttl_s: int = _DEFAULT_TTL_S) -> None:
        self._redis.set(key, value, ex=ttl_s)

    def get(self, key: str) -> str | None:
        return self._redis.get(key)

    def delete(self, key: str) -> None:
        self._redis.delete(key)

    def exists(self, key: str) -> bool:
        return self._redis.exists(key) == 1


_store: ProgressStore | None = None


def get_store() -> ProgressStore:
    global _store
    if _store is None:
        # `celery` runtime uses Redis so API and worker processes share it;
        # `local` (desktop) and the test suite stay in-process.
        if settings.ALF_TASK_QUEUE == "celery" and settings.ENV != "test":
            _store = RedisStore(settings.REDIS_URL)
        else:
            _store = InMemoryStore()
    return _store
