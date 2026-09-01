"""Live training progress — same pattern as `workers/progress.py` but shaped
for per-epoch metrics rather than per-image counts. Kept as a separate
module since the payload shape (mAP/precision/recall vs fps/predictions) is
genuinely different, not because the mechanism is. Backing store is
in-process (desktop) or Redis (server); see `workers/progress_store.py`."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from app.workers.progress_store import get_store

_TTL_S = 3600


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
    get_store().set(_key(job_id), json.dumps(asdict(progress)), _TTL_S)


def get_training_progress(job_id: str) -> EpochProgress | None:
    raw = get_store().get(_key(job_id))
    if raw is None:
        return None
    return EpochProgress(**json.loads(raw))
