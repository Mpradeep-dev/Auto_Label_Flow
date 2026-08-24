"""TrainingProvider — the interface every training backend sits behind
(PLAN "Training Provider Abstraction"). The application calls
`training_provider.start_training(job)` and never branches on LOCAL vs
KAGGLE outside `registry.py` — adding AWS/Azure/GCP later is one new
module plus one registry line, not a change to any caller.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from sqlalchemy.orm import Session

from app.models.training_job import TrainingJob


class TrainingProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """False means this provider can't run right now (e.g. no Kaggle
        credentials). The application must degrade gracefully, not crash —
        see PLAN 'the unconfigured path is the default path'."""

    @abstractmethod
    def start_training(self, db: Session, job: TrainingJob) -> None:
        """Kick off training for `job` (already persisted with QUEUED
        status). Implementations own how the work actually runs — a local
        Celery task, a remote API call — and are responsible for updating
        `job.status`/`job.error`/etc. as it progresses."""

    @abstractmethod
    def get_status(self, db: Session, job: TrainingJob) -> str:
        """Returns the current TrainingJobStatus value. Implementations
        may refresh remote state (e.g. poll Kaggle) before returning."""

    @abstractmethod
    def get_logs(self, db: Session, job: TrainingJob) -> str: ...

    @abstractmethod
    def download_artifacts(self, db: Session, job: TrainingJob) -> Path | None:
        """Path to the trained weights once available, else None."""

    @abstractmethod
    def cancel_training(self, db: Session, job: TrainingJob) -> None: ...


class ModelIdRequiredError(ValueError):
    pass
