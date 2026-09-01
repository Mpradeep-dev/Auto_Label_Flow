"""Live progress for jobs, read by the SSE endpoint (PLAN "Progress ...
written on a ~0.5s throttle and streamed over SSE"). The DB row
(`inference_jobs`) is the durable checkpoint at batch boundaries — this
module is the fast, ephemeral path the UI polls.

The backing store is an in-process dict on the desktop app and Redis on the
server; see `workers/progress_store.py`. This module's public API is
unchanged."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

from app.workers.progress_store import get_store

_PROGRESS_TTL_S = 3600
_THROTTLE_S = 0.5


@dataclass
class JobProgress:
    current: int
    total: int
    predictions: int = 0
    fps: float = 0.0
    eta_s: float | None = None
    status: str = "RUNNING"
    error: str | None = None


def _key(job_id: str) -> str:
    return f"job:progress:{job_id}"


def _cancel_key(job_id: str) -> str:
    return f"job:cancel:{job_id}"


class ThrottledProgressWriter:
    """Wraps the Redis write in a per-call time throttle so a tight
    per-image loop doesn't hammer Redis — writes are cheap but not free at
    thousands of images/sec of loop overhead."""

    def __init__(self, job_id: str, total: int) -> None:
        self.job_id = job_id
        self.total = total
        self._start = time.monotonic()
        self._last_write = 0.0
        self._current = 0
        self._predictions = 0

    def update(self, current: int, predictions_delta: int = 0, force: bool = False) -> None:
        self._current = current
        self._predictions += predictions_delta
        now = time.monotonic()
        if not force and (now - self._last_write) < _THROTTLE_S:
            return
        self._last_write = now
        elapsed = max(now - self._start, 1e-6)
        fps = current / elapsed
        remaining = self.total - current
        eta = remaining / fps if fps > 0 else None
        set_progress(
            self.job_id,
            JobProgress(current=current, total=self.total, predictions=self._predictions, fps=fps, eta_s=eta),
        )

    def finish(self, status: str = "COMPLETED", error: str | None = None) -> None:
        set_progress(
            self.job_id,
            JobProgress(
                current=self._current,
                total=self.total,
                predictions=self._predictions,
                fps=0.0,
                eta_s=0.0,
                status=status,
                error=error,
            ),
        )


def set_progress(job_id: str, progress: JobProgress) -> None:
    get_store().set(_key(job_id), json.dumps(asdict(progress)), _PROGRESS_TTL_S)


def get_progress(job_id: str) -> JobProgress | None:
    raw = get_store().get(_key(job_id))
    if raw is None:
        return None
    return JobProgress(**json.loads(raw))


def request_cancel(job_id: str) -> None:
    get_store().set(_cancel_key(job_id), "1", _PROGRESS_TTL_S)


def is_cancel_requested(job_id: str) -> bool:
    return get_store().exists(_cancel_key(job_id))


def clear_cancel(job_id: str) -> None:
    get_store().delete(_cancel_key(job_id))
