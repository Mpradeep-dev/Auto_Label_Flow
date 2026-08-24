"""Redis-backed live training progress — same pattern as
`workers/progress.py` but shaped for per-epoch metrics rather than
per-image counts. Kept as a separate module since the payload shape
(mAP/precision/recall vs fps/predictions) is genuinely different, not
because the mechanism is."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import redis

from app.core.config import settings

_TTL_S = 3600
_redis = redis.from_url(settings.REDIS_URL, decode_responses=True)


@dataclass
class EpochProgress:
    epoch: int
    total_epochs: int
    box_loss: float | None = None
    cls_loss: float | None = None
    dfl_loss: float | None = None
    precision: float | None = None
    recall: float | None = None
    map50: float | None = None
    map50_95: float | None = None
    status: str = "RUNNING"
    error: str | None = None


def _key(job_id: str) -> str:
    return f"training:progress:{job_id}"


def set_training_progress(job_id: str, progress: EpochProgress) -> None:
    _redis.set(_key(job_id), json.dumps(asdict(progress)), ex=_TTL_S)


def get_training_progress(job_id: str) -> EpochProgress | None:
    raw = _redis.get(_key(job_id))
    if raw is None:
        return None
    return EpochProgress(**json.loads(raw))
