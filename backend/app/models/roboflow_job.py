"""Roboflow import/export job — the background-job counterpart of
`InferenceJob`/`TrainingJob`, added so the Settings-page "connect Roboflow"
promise ("both actions live on the Dataset and Export pages") has something
a progress bar can poll. Import and export share one table (like
`TrainingJob` mixes provider-specific nullable columns) rather than two,
since they differ only in which handful of fields are populated — `kind`
picks the branch.
"""
from __future__ import annotations

import uuid
from enum import Enum as PyEnum

from sqlalchemy import Boolean, Enum, ForeignKey, Integer, JSON, String
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class RoboflowJobKind(str, PyEnum):
    IMPORT = "IMPORT"
    EXPORT = "EXPORT"


class RoboflowJobStatus(str, PyEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class RoboflowJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "roboflow_jobs"

    project_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[RoboflowJobKind] = mapped_column(Enum(RoboflowJobKind, name="roboflow_job_kind"), nullable=False)
    status: Mapped[RoboflowJobStatus] = mapped_column(
        Enum(RoboflowJobStatus, name="roboflow_job_status"),
        nullable=False,
        default=RoboflowJobStatus.QUEUED,
        index=True,  # DB-05: the natural filter for "show running/queued jobs" UI polling
    )

    workspace: Mapped[str] = mapped_column(String(200), nullable=False)
    project_slug: Mapped[str] = mapped_column(String(200), nullable=False)

    # IMPORT only
    version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dataset_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    # Only meaningful for a raw pull (version is None) — narrows it to
    # images with zero existing annotations, instead of every raw image.
    unannotated_only: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    result_dataset_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True
    )

    # EXPORT only
    dataset_version_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("dataset_versions.id", ondelete="SET NULL"), nullable=True
    )
    uploaded_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failures: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    total_items: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processed_items: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    celery_task_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    error: Mapped[str | None] = mapped_column(String(2000), nullable=True)
