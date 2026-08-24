"""Training job — the loop-closing table. `provider` decides which
TrainingProvider implementation runs it (PLAN "Training providers");
`result_model_id` is set on completion and is what lets the newly-trained
model immediately become selectable for auto-annotation, same registry as
every other model (`app/services/inference/registry.py`)."""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class TrainingProviderType(str, PyEnum):
    LOCAL = "LOCAL"
    KAGGLE = "KAGGLE"


class TrainingJobStatus(str, PyEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TrainingJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "training_jobs"

    project_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    dataset_version_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("dataset_versions.id", ondelete="RESTRICT"), nullable=False
    )
    base_model_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("models.id", ondelete="SET NULL"), nullable=True
    )
    result_model_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("models.id", ondelete="SET NULL"), nullable=True
    )

    provider: Mapped[TrainingProviderType] = mapped_column(
        Enum(TrainingProviderType, name="training_provider_type"), nullable=False
    )
    status: Mapped[TrainingJobStatus] = mapped_column(
        Enum(TrainingJobStatus, name="training_job_status"), nullable=False, default=TrainingJobStatus.QUEUED
    )

    epochs: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False, default=8)
    image_size: Mapped[int] = mapped_column(Integer, nullable=False, default=640)
    learning_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    device: Mapped[str] = mapped_column(String(20), nullable=False, default="cpu")

    current_epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # final metrics snapshot

    celery_task_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    kaggle_kernel_ref: Mapped[str | None] = mapped_column(String(300), nullable=True)
    artifact_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    error: Mapped[str | None] = mapped_column(String(2000), nullable=True)

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    failed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class TrainingJobEpoch(UUIDPrimaryKeyMixin, Base):
    """One row per completed epoch — a separate table (not a JSON array on
    TrainingJob) specifically so the frontend can chart live loss/mAP with
    a plain indexed query instead of re-parsing a growing JSON blob on
    every poll."""

    __tablename__ = "training_job_epochs"

    training_job_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("training_jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    box_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    cls_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    dfl_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    map50: Mapped[float | None] = mapped_column(Float, nullable=True)
    map50_95: Mapped[float | None] = mapped_column(Float, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
