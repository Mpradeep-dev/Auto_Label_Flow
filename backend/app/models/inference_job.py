"""Inference job — tracks one batch auto-annotation run over a dataset.
The durable checkpoint (PLAN: "DB row checkpointed at batch boundaries");
live progress between checkpoints lives in Redis (`workers/progress.py`)."""
from __future__ import annotations

import uuid
from enum import Enum as PyEnum

from sqlalchemy import Enum, Float, ForeignKey, Index, Integer, String, text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class JobStatus(str, PyEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class InferenceJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "inference_jobs"
    __table_args__ = (
        # At most one QUEUED/RUNNING job per dataset (audit finding BE-03) —
        # a partial unique index so job *history* is unlimited but two jobs
        # can never be in flight against the same dataset at once, closing
        # the race that an app-level check-then-insert can't. Declared here
        # (not just in the matching Alembic migration,
        # e7c2a4f691b5_one_active_inference_job_per_dataset) because the
        # test suite builds its schema straight from this metadata rather
        # than running migrations — see tests/conftest.py.
        Index(
            "uq_inference_jobs_one_active_per_dataset",
            "dataset_id",
            unique=True,
            postgresql_where=text("status IN ('QUEUED', 'RUNNING')"),
        ),
    )

    project_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("models.id", ondelete="RESTRICT"), nullable=False
    )

    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status"), nullable=False, default=JobStatus.QUEUED, index=True
    )
    conf: Mapped[float] = mapped_column(Float, nullable=False, default=0.20)
    iou: Mapped[float] = mapped_column(Float, nullable=False, default=0.70)

    total_images: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processed_images: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_images: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_predictions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    celery_task_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    error: Mapped[str | None] = mapped_column(String(2000), nullable=True)
