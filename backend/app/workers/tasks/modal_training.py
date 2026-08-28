"""Modal training: job submission (async) + periodic status poll.

Mirrors the Kaggle training pattern (kaggle_training.py):
- start_modal_training_job: packages dataset, uploads to Modal Volume,
  submits training function to Modal's GPU cloud
- poll_modal_training_jobs: periodic beat task that checks status and
  finalizes completed jobs (downloads best.pt, registers model)
"""
from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.ml_model import MLModel, ModelKind
from app.models.training_job import TrainingJob, TrainingJobStatus, TrainingProviderType
from app.services.inference.registry import register_model
from app.services.training.modal_provider import ModalTrainingProvider
from app.workers.celery_app import (
    MODAL_START_SOFT_TIME_LIMIT_S,
    MODAL_START_TIME_LIMIT_S,
    celery_app,
)


@celery_app.task(
    name="app.workers.tasks.modal_training.start_modal_training_job",
    time_limit=MODAL_START_TIME_LIMIT_S,
    soft_time_limit=MODAL_START_SOFT_TIME_LIMIT_S,
)
def start_modal_training_job(job_id: str) -> None:
    db = SessionLocal()
    try:
        job = db.get(TrainingJob, uuid.UUID(job_id))
        if job is None:
            return
        provider = ModalTrainingProvider()
        try:
            provider._submit_job(db, job)
        except Exception as exc:
            db.rollback()
            job = db.get(TrainingJob, uuid.UUID(job_id))
            job.status = TrainingJobStatus.FAILED
            job.error = str(exc)[:2000]
            job.failed_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        db.close()


def _finalize_completed_job(db: Session, provider: ModalTrainingProvider, job: TrainingJob) -> None:
    """Pull best.pt from Modal Volume, register as a new model — same
    pattern as kaggle_training.py's _finalize_completed_job."""
    base_model = db.get(MLModel, job.base_model_id) if job.base_model_id else None
    try:
        weights_path = provider.download_artifacts(db, job)
        if weights_path is None or base_model is None:
            job.status = TrainingJobStatus.FAILED
            job.error = "Modal training completed but no best.pt weights file was found."
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
            version=f"trained-from-{base_model.name}-{str(job.id)[:8]}",
        )
        result_model.base_model_id = base_model.id
        job.result_model_id = result_model.id
        job.artifact_path = str(registered_path)
        job.completed_at = job.completed_at or datetime.now(timezone.utc)
        db.commit()

        # Clean up the Modal Volume after successful download
        try:
            import modal

            volume_name = f"annotate-training-{str(job.id)[:12]}"
            client = provider._get_client()
            vol = modal.Volume.from_name(volume_name, client=client)
            vol.delete()
        except Exception:
            pass  # best-effort cleanup
    except Exception as exc:
        db.rollback()
        job = db.get(TrainingJob, job.id)
        job.status = TrainingJobStatus.FAILED
        job.error = f"Failed to register the Modal training result: {exc}"[:2000]
        job.failed_at = datetime.now(timezone.utc)
        db.commit()


@celery_app.task(name="app.workers.tasks.modal_training.poll_modal_training_jobs")
def poll_modal_training_jobs() -> dict[str, int]:
    """Periodic task: poll Modal for job status updates.

    Unlike Kaggle (which has an explicit kernels_status API), Modal's
    .spawn() returns a FunctionCall reference. We check completion by
    attempting a non-blocking get() on the reference.

    For simplicity in v1, we use a time-based heuristic: if the job has
    been running longer than expected, check the Volume for best.pt.
    A future improvement could use Modal's FunctionCall.get_status() API.
    """
    db = SessionLocal()
    counts = {"polled": 0, "completed": 0, "failed": 0}
    provider = ModalTrainingProvider()
    try:
        jobs = list(
            db.query(TrainingJob).filter(
                TrainingJob.provider == TrainingProviderType.MODAL,
                TrainingJob.status.in_([TrainingJobStatus.QUEUED, TrainingJobStatus.RUNNING]),
            )
        )
        for job in jobs:
            counts["polled"] += 1
            try:
                # Heartbeat: keep the job alive in reconcile's eyes
                job.updated_at = datetime.now(timezone.utc)
                db.commit()

                # Check if best.pt exists in the Volume
                volume_name = f"annotate-training-{str(job.id)[:12]}"
                import modal

                client = provider._get_client()
                try:
                    vol = modal.Volume.from_name(volume_name, client=client)
                    vol.reload()
                    # Check if best.pt exists in the volume root
                    files = [entry for entry in vol.listdir("/") if entry.path == "best.pt"]
                    if files:
                        # Training complete — finalize
                        job.status = TrainingJobStatus.COMPLETED
                        job.completed_at = datetime.now(timezone.utc)
                        db.commit()
                        _finalize_completed_job(db, provider, job)
                        counts["completed"] += 1
                        continue
                except Exception:
                    pass

                # If the job has been running for more than 6 hours, fail it
                if job.started_at:
                    elapsed = (datetime.now(timezone.utc) - job.started_at).total_seconds()
                    if elapsed > 6 * 3600:
                        job.status = TrainingJobStatus.FAILED
                        job.error = "Modal training job timed out after 6 hours"
                        job.failed_at = datetime.now(timezone.utc)
                        db.commit()
                        counts["failed"] += 1

            except Exception as exc:
                db.rollback()
                job = db.get(TrainingJob, job.id)
                if job is not None:
                    job.error = f"Modal status check failed: {exc}"[:2000]
                    db.commit()
                continue

        return counts
    finally:
        db.close()
