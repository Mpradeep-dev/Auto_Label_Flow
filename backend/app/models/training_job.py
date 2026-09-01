"""Training job — the loop-closing table. `provider` decides which
TrainingProvider implementation runs it (PLAN "Training providers");
`result_model_id` is set on completion and is what lets the newly-trained
model immediately become selectable for auto-annotation, same registry as
every other model (`app/services/inference/registry.py`)."""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import Boolean, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.db.types import GUID, TZDateTime, enum_column


class TrainingProviderType(str, PyEnum):
    LOCAL = "LOCAL"
    KAGGLE = "KAGGLE"
    MODAL = "MODAL"


class TrainingJobStatus(str, PyEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TrainingJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "training_jobs"

    project_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # CASCADE, not RESTRICT: there is no standalone "delete this dataset
    # version" endpoint in this app — the only ways a dataset_versions row
    # is ever deleted are cascading from `DELETE /datasets/{id}` or
    # `DELETE /projects/{id}`, both of which mean to take everything
    # underneath with them (training runs included). RESTRICT here only
    # ever blocked those legitimate cascades — confirmed live: deleting a
    # project with a training job on one of its versions 500'd with
    # `ForeignKeyViolation: training_jobs_dataset_version_id_fkey`.
    dataset_version_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False
    )
    base_model_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID, ForeignKey("models.id", ondelete="SET NULL"), nullable=True
    )
    result_model_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID, ForeignKey("models.id", ondelete="SET NULL"), nullable=True
    )
    # User-supplied name for the model this job produces. Every provider's
    # finalize step (train_local_model, kaggle_training._finalize_completed_job,
    # modal_training._finalize_completed_job) used to hardcode
    # f"{base_model.name}-retrained" unconditionally — fine for one run, but
    # indistinguishable once you've retrained the same base model more than
    # once (the Models list shows N identical "detect_v1-retrained" rows,
    # told apart only by an opaque version string). NULL falls back to that
    # same default name, so this is purely additive.
    result_model_name: Mapped[str | None] = mapped_column(String(200), nullable=True)

    provider: Mapped[TrainingProviderType] = mapped_column(
        enum_column(TrainingProviderType, "training_provider_type"), nullable=False
    )
    status: Mapped[TrainingJobStatus] = mapped_column(
        enum_column(TrainingJobStatus, "training_job_status"),
        nullable=False,
        default=TrainingJobStatus.QUEUED,
        index=True,  # DB-05: the natural filter for "show running/queued jobs" UI polling
    )

    epochs: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False, default=8)
    image_size: Mapped[int] = mapped_column(Integer, nullable=False, default=640)
    learning_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    device: Mapped[str] = mapped_column(String(20), nullable=False, default="cpu")
    # KAGGLE-only (the LOCAL provider ignores this — it always uses whatever
    # `device` resolves to, since it's your own machine). Kaggle accounts
    # have a weekly GPU-hours quota; before this, kaggle_provider.py always
    # requested a GPU kernel unconditionally, with no way to run a
    # quota-free CPU kernel instead. Defaults True to match that prior
    # always-on behavior for anyone who doesn't touch this.
    enable_gpu: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    # Passthrough for any Ultralytics `YOLO.train()` keyword not already a
    # typed column above (optimizer, patience, dropout, augmentation knobs,
    # etc.) — Ultralytics has ~100 of these; a dedicated column per one
    # isn't worth it, and the typed columns above win on conflict (see
    # `train_local_model`'s merge order) so this can never smuggle in a
    # different epochs/imgsz/batch/device than the job row actually records.
    extra_args: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    current_epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # final metrics snapshot

    celery_task_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    kaggle_kernel_ref: Mapped[str | None] = mapped_column(String(300), nullable=True)
    modal_function_call_id: Mapped[str | None] = mapped_column(String(300), nullable=True)
    artifact_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    error: Mapped[str | None] = mapped_column(String(2000), nullable=True)

    started_at: Mapped[datetime | None] = mapped_column(TZDateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(TZDateTime, nullable=True)
    failed_at: Mapped[datetime | None] = mapped_column(TZDateTime, nullable=True)


class TrainingJobEpoch(UUIDPrimaryKeyMixin, Base):
    """One row per completed epoch — a separate table (not a JSON array on
    TrainingJob) specifically so the frontend can chart live loss/mAP with
    a plain indexed query instead of re-parsing a growing JSON blob on
    every poll."""

    __tablename__ = "training_job_epochs"

    training_job_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("training_jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    box_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    cls_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    dfl_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    map50: Mapped[float | None] = mapped_column(Float, nullable=True)
    map50_95: Mapped[float | None] = mapped_column(Float, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(TZDateTime, nullable=False)
