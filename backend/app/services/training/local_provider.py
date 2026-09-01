from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from app.models.training_job import TrainingJob, TrainingJobStatus
from app.services.training.provider import TrainingProvider
from app.workers.progress import request_cancel  # shared cancel-flag mechanism (progress-store seam)
from app.workers.training_progress import get_training_progress


class LocalTrainingProvider(TrainingProvider):
    @property
    def name(self) -> str:
        return "LOCAL"

    @property
    def is_configured(self) -> bool:
        # Dev / server: torch is already installed however the operator wants
        # it, so local training has nothing to configure. Packaged desktop
        # app: the base install is CPU-only, so local GPU YOLO training needs
        # the optional GPU pack (CUDA torch) downloaded from Settings first.
        from app.core.config import settings

        if not settings.ALF_DATA_DIR:
            return True
        from app.services.system import packs

        return packs.is_installed("gpu")

    def start_training(self, db: Session, job: TrainingJob) -> None:
        from app.workers.tasks.training import train_local_model

        train_local_model.delay(str(job.id))

    def get_status(self, db: Session, job: TrainingJob) -> str:
        return job.status.value

    def get_logs(self, db: Session, job: TrainingJob) -> str:
        progress = get_training_progress(str(job.id))
        if progress is None:
            return f"status={job.status.value}"
        return f"epoch {progress.epoch}/{progress.total_epochs} — {progress.status}" + (
            f" — {progress.error}" if progress.error else ""
        )

    def download_artifacts(self, db: Session, job: TrainingJob) -> Path | None:
        return Path(job.artifact_path) if job.artifact_path else None

    def cancel_training(self, db: Session, job: TrainingJob) -> None:
        # Checked in the `on_fit_epoch_end` callback (workers/tasks/training.py),
        # which sets `trainer.stop = True` — Ultralytics' BaseTrainer checks
        # this after each epoch and breaks its loop there. Same "checked
        # between units of work" pattern as batch inference's cancel flag,
        # just at epoch granularity rather than per-image: there's no finer
        # hook into a single `.train()` call to interrupt mid-epoch.
        request_cancel(str(job.id))
