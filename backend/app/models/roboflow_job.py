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

from sqlalchemy import Boolean, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.db.types import GUID, enum_column


class RoboflowJobKind(str, PyEnum):
    IMPORT = "IMPORT"
    EXPORT = "EXPORT"


class RoboflowJobStatus(str, PyEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class RoboflowUploadTarget(str, PyEnum):
    """EXPORT only — which column of Roboflow's Annotate board a pushed
    image should land in. `ANNOTATING` (the historical, still-default,
    behavior) sends any local label as a *prediction*, which Roboflow
    queues for review; `DATASET` sends it as ground truth instead, which
    Roboflow auto-confirms straight into the Dataset column; `UNANNOTATED`
    strips labels before upload regardless of what's local, so the image
    lands with no annotation at all. An image with no local label always
    lands unannotated no matter the target — there's nothing to push as a
    prediction or ground truth for it."""

    UNANNOTATED = "UNANNOTATED"
    ANNOTATING = "ANNOTATING"
    DATASET = "DATASET"


class RoboflowJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "roboflow_jobs"

    project_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[RoboflowJobKind] = mapped_column(enum_column(RoboflowJobKind, "roboflow_job_kind"), nullable=False)
    status: Mapped[RoboflowJobStatus] = mapped_column(
        enum_column(RoboflowJobStatus, "roboflow_job_status"),
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
    # Only meaningful for a raw pull (version is None) — narrows it to one
    # upload batch (Roboflow's own batch id), instead of every raw image.
    batch_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    # Applies to both versioned and raw pulls — the image is still pulled
    # and stored, but whatever labels Roboflow already has on it are not
    # imported, so it lands with zero annotations for auto-annotate to
    # start fresh on (issue #22: existing Roboflow labels get in the way).
    images_only: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    result_dataset_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID, ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True
    )

    # EXPORT only
    dataset_version_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID, ForeignKey("dataset_versions.id", ondelete="SET NULL"), nullable=True
    )
    # User-chosen upload batch label — overrides the auto-generated
    # `AutoLabelFlow-{dataset}-v{n}` one when set (see roboflow_export.py's
    # push_version_to_roboflow). None means "use the default".
    batch_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    # Which Roboflow Annotate-board column pushed images land in — see
    # `RoboflowUploadTarget`. Plain `String`, not `enum_column`, matching
    # this table's other EXPORT-only columns (`batch_name`) rather than the
    # native-Postgres-enum treatment `kind`/`status` get, since this value
    # is never queried/filtered on and doesn't need a DB-level CHECK.
    upload_target: Mapped[str] = mapped_column(
        String(20), nullable=False, default=RoboflowUploadTarget.ANNOTATING.value
    )
    uploaded_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failures: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    total_items: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processed_items: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    celery_task_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    error: Mapped[str | None] = mapped_column(String(2000), nullable=True)
