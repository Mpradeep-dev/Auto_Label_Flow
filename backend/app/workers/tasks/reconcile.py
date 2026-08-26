"""Stale-job reconciliation (audit finding REL-01, second half).

Nothing else in this codebase notices when a job's *worker* dies mid-task —
a Docker restart, an OOM kill, a hard Celery time-limit kill — rather than
the task itself failing in a way its own `except Exception` can catch. Any
of those leaves the job's DB row sitting at QUEUED/RUNNING forever: no
heartbeat, no timeout, nothing. The reattach-on-reload UI (Auto Annotation,
Training Runs) then polls a job that will never move again, with no way to
tell "still working" from "silently dead."

This runs on a schedule (see `celery_app.py`'s `beat_schedule`) rather than
relying on the crashed task to report its own death, which is the whole
point — it doesn't depend on the thing that failed getting a chance to
clean up after itself.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.db.session import SessionLocal
from app.models.inference_job import InferenceJob, JobStatus as InferenceJobStatus
from app.models.roboflow_job import RoboflowJob, RoboflowJobStatus
from app.models.training_job import TrainingJob, TrainingJobStatus
from app.models.video import Video, VideoStatus
from app.workers.celery_app import RECONCILE_STALE_AFTER_S, celery_app

_STALE_MESSAGE = (
    "No progress for longer than expected — the worker likely crashed or was restarted "
    "mid-job. Marked failed automatically; try again."
)


@celery_app.task(name="app.workers.tasks.reconcile.reconcile_stale_jobs")
def reconcile_stale_jobs() -> dict[str, int]:
    db = SessionLocal()
    counts = {"inference": 0, "training": 0, "video": 0, "roboflow": 0}
    try:
        now = datetime.now(timezone.utc)

        cutoff = now - timedelta(seconds=RECONCILE_STALE_AFTER_S["inference"])
        for job in db.query(InferenceJob).filter(
            InferenceJob.status.in_([InferenceJobStatus.QUEUED, InferenceJobStatus.RUNNING]),
            InferenceJob.updated_at < cutoff,
        ):
            job.status = InferenceJobStatus.FAILED
            job.error = _STALE_MESSAGE
            counts["inference"] += 1

        cutoff = now - timedelta(seconds=RECONCILE_STALE_AFTER_S["training"])
        for job in db.query(TrainingJob).filter(
            TrainingJob.status.in_([TrainingJobStatus.QUEUED, TrainingJobStatus.RUNNING]),
            TrainingJob.updated_at < cutoff,
        ):
            job.status = TrainingJobStatus.FAILED
            job.error = _STALE_MESSAGE
            counts["training"] += 1

        cutoff = now - timedelta(seconds=RECONCILE_STALE_AFTER_S["video"])
        for video in db.query(Video).filter(Video.status == VideoStatus.EXTRACTING, Video.updated_at < cutoff):
            video.status = VideoStatus.FAILED
            video.error = _STALE_MESSAGE
            counts["video"] += 1

        cutoff = now - timedelta(seconds=RECONCILE_STALE_AFTER_S["roboflow"])
        for job in db.query(RoboflowJob).filter(
            RoboflowJob.status.in_([RoboflowJobStatus.QUEUED, RoboflowJobStatus.RUNNING]),
            RoboflowJob.updated_at < cutoff,
        ):
            job.status = RoboflowJobStatus.FAILED
            job.error = _STALE_MESSAGE
            counts["roboflow"] += 1

        db.commit()
        return counts
    finally:
        db.close()
