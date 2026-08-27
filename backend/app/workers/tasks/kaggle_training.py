"""Kaggle training: kernel push (async) + periodic status poll.

A LOCAL job's own Celery task updates its status/epoch/metrics as it runs
(`workers/tasks/training.py`, `on_fit_epoch_end`). A Kaggle job trains on
Kaggle's own remote infrastructure, entirely outside this app's process —
`KaggleTrainingProvider._push_kernel()` just pushes a kernel and returns.
Before the poll task below existed, nothing ever called `get_status`/
`download_artifacts` afterward, so a Kaggle job sat frozen at
RUNNING/epoch 0 forever: not because the Kaggle kernel was actually stuck,
but because this app never once asked Kaggle how it was doing. The only
thing that eventually touched it was `tasks/reconcile.py`'s stale-job
sweep, hours later, misreporting a healthy-but-unpolled job as a crashed
one.

`poll_kaggle_training_jobs` is that fix: on a schedule (see
`celery_app.py`'s `beat_schedule`), poll every Kaggle job still
QUEUED/RUNNING, and on completion mirror `train_local_model`'s own
finalization — pull the trained weights, register them as a new model,
close the loop the same way LOCAL training does.

`start_kaggle_training_job` is a separate, later fix: `_push_kernel` itself
(dataset export/zip, upload, a readiness poll that alone can take up to 5
minutes, kernel push) used to run synchronously inside `POST
/training/jobs`'s request handler — confirmed live to read, from the
frontend, as "no progress update, then eventually just fails": the whole
call is one blocking round trip, so the UI has nothing to show but a
static "Starting…" for however long Kaggle's side takes. This task moves
that work off the request thread, the same way LOCAL training already
does via `train_local_model.delay(...)` — `POST /training/jobs` now
returns a QUEUED row immediately, and the frontend's existing
QUEUED/RUNNING polling (`TrainingRunsPage.tsx`) picks up real progress
once this task actually gets the kernel running.
"""
from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.ml_model import MLModel, ModelKind
from app.models.training_job import TrainingJob, TrainingJobEpoch, TrainingJobStatus, TrainingProviderType
from app.services.inference.registry import register_model
from app.services.training.kaggle_provider import KaggleTrainingProvider
from app.services.training.ultralytics_log_parser import parse_ultralytics_epochs
from app.workers.celery_app import (
    KAGGLE_START_SOFT_TIME_LIMIT_S,
    KAGGLE_START_TIME_LIMIT_S,
    celery_app,
)


@celery_app.task(
    name="app.workers.tasks.kaggle_training.start_kaggle_training_job",
    time_limit=KAGGLE_START_TIME_LIMIT_S,
    soft_time_limit=KAGGLE_START_SOFT_TIME_LIMIT_S,
)
def start_kaggle_training_job(job_id: str) -> None:
    db = SessionLocal()
    try:
        job = db.get(TrainingJob, uuid.UUID(job_id))
        if job is None:
            return
        provider = KaggleTrainingProvider()
        try:
            provider._push_kernel(db, job)
        except Exception as exc:
            # Same clean-FAILED-state handling POST /training/jobs used to
            # do inline (see audit finding BE-01 in that endpoint's own
            # history) — now here, since the request thread returned long
            # before this could possibly raise.
            db.rollback()
            job = db.get(TrainingJob, uuid.UUID(job_id))
            job.status = TrainingJobStatus.FAILED
            job.error = str(exc)[:2000]
            job.failed_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        db.close()


def _finalize_completed_job(db: Session, provider: KaggleTrainingProvider, job: TrainingJob) -> None:
    """Mirrors `train_local_model`'s completion block (workers/tasks/
    training.py): pull the best.pt Kaggle produced, register it as a new
    model, wire up `result_model_id` — the same loop-closing step LOCAL
    training already does, just fed from a downloaded kernel output
    instead of a local training-run directory."""
    base_model = db.get(MLModel, job.base_model_id) if job.base_model_id else None
    try:
        weights_path = provider.download_artifacts(db, job)
        if weights_path is None or base_model is None:
            job.status = TrainingJobStatus.FAILED
            job.error = "Kaggle kernel completed but no best.pt weights file was found in its output."
            job.failed_at = datetime.now(timezone.utc)
            db.commit()
            return

        registered_path = settings.MODELS_DIR / "pt" / f"trained-{job.id}.pt"
        registered_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(weights_path, registered_path)

        result_model = register_model(
            db,
            name=job.result_model_name or f"{base_model.name}-retrained",
            weights_path=str(registered_path),
            kind=ModelKind.DETECTOR,
            # Same collision-avoidance reasoning as train_local_model's own
            # version string — two Kaggle runs off the same base model
            # would otherwise produce identical (name, version) pairs.
            version=f"trained-from-{base_model.name}-{str(job.id)[:8]}",
        )
        result_model.base_model_id = base_model.id
        job.result_model_id = result_model.id
        job.artifact_path = str(registered_path)
        job.completed_at = job.completed_at or datetime.now(timezone.utc)
        db.commit()
    except Exception as exc:
        db.rollback()
        job = db.get(TrainingJob, job.id)
        job.status = TrainingJobStatus.FAILED
        job.error = f"Failed to register the Kaggle training result: {exc}"[:2000]
        job.failed_at = datetime.now(timezone.utc)
        db.commit()


def _sync_epoch_progress(db: Session, provider: KaggleTrainingProvider, job: TrainingJob) -> None:
    """Best-effort live progress for a RUNNING Kaggle job: pull the kernel's
    console log and regex out any epoch rows LOCAL training would otherwise
    get from Ultralytics' `on_fit_epoch_end` callback directly (see
    ultralytics_log_parser.py for why this is inherently best-effort, not
    guaranteed). Never raises — a parse miss just means no new rows this
    poll, not a failed job; the caller doesn't need its own try/except."""
    try:
        log_text = provider.get_logs(db, job)
    except Exception:
        return  # log not fetchable yet (e.g. kernel just started) — try again next poll

    try:
        epochs = parse_ultralytics_epochs(log_text)
    except Exception:
        return  # a log format this parser doesn't recognize — degrade to no epoch history, not a crash

    new_epochs = [e for e in epochs if e["epoch"] > job.current_epoch]
    if not new_epochs:
        return

    for e in new_epochs:
        db.add(
            TrainingJobEpoch(
                training_job_id=job.id,
                epoch=e["epoch"],
                box_loss=e["box_loss"],
                cls_loss=e["cls_loss"],
                dfl_loss=e["dfl_loss"],
                precision=e["precision"],
                recall=e["recall"],
                map50=e["map50"],
                map50_95=e["map50_95"],
                recorded_at=datetime.now(timezone.utc),
            )
        )
    latest = new_epochs[-1]
    job.current_epoch = latest["epoch"]
    job.metrics = {k: v for k, v in latest.items() if k != "epoch" and v is not None}
    db.commit()


@celery_app.task(name="app.workers.tasks.kaggle_training.poll_kaggle_training_jobs")
def poll_kaggle_training_jobs() -> dict[str, int]:
    db = SessionLocal()
    counts = {"polled": 0, "completed": 0, "failed": 0}
    provider = KaggleTrainingProvider()
    try:
        jobs = list(
            db.query(TrainingJob).filter(
                TrainingJob.provider == TrainingProviderType.KAGGLE,
                TrainingJob.status.in_([TrainingJobStatus.QUEUED, TrainingJobStatus.RUNNING]),
            )
        )
        for job in jobs:
            counts["polled"] += 1
            try:
                status = provider.get_status(db, job)
                # Heartbeat: touched on every successful poll, even when
                # status didn't change, so a long-but-legitimately-running
                # Kaggle job doesn't get killed by the stale-job sweep
                # (tasks/reconcile.py) — that sweep only sees `updated_at`,
                # and without this, polling itself (not the job) would be
                # the only thing that could have kept it moving.
                job.updated_at = datetime.now(timezone.utc)
                db.commit()
            except Exception as exc:
                db.rollback()
                job = db.get(TrainingJob, job.id)
                if job is not None:
                    job.error = f"Kaggle status check failed: {exc}"[:2000]
                    db.commit()
                continue

            if status == TrainingJobStatus.COMPLETED.value and job.result_model_id is None:
                _finalize_completed_job(db, provider, job)
                counts["completed"] += 1
            elif status == TrainingJobStatus.RUNNING.value:
                _sync_epoch_progress(db, provider, job)
            elif status == TrainingJobStatus.FAILED.value and not job.error:
                try:
                    job.error = provider.get_logs(db, job)[-2000:]
                    db.commit()
                except Exception:
                    pass  # best-effort — a FAILED status is still recorded even if the log fetch itself fails
                counts["failed"] += 1
        return counts
    finally:
        db.close()
