"""Azure-Blob import job — the background-job row a progress bar can poll
while `services/integrations/azure_blob_import.py` walks a container prefix
and registers its images. Deliberately a separate, minimal table rather
than another `kind` on `roboflow_jobs`: the two share no columns beyond
the generic job scaffolding, and folding an Azure concept into a table
named `roboflow_jobs` would just be misleading.

Same progress/cancel mechanism as every other job (`app.workers.progress`
+ an SSE `/stream` endpoint), so the frontend polls it exactly like a
Roboflow import job.
"""
from __future__ import annotations

import uuid
from enum import Enum as PyEnum

from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.db.types import GUID, enum_column


class BlobImportJobStatus(str, PyEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class BlobImportJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "blob_import_jobs"

    project_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[BlobImportJobStatus] = mapped_column(
        enum_column(BlobImportJobStatus, "blob_import_job_status"),
        nullable=False,
        default=BlobImportJobStatus.QUEUED,
        index=True,
    )

    # Blob-name prefix inside the app's configured AZURE_STORAGE_CONTAINER
    # to pull from, e.g. "prod-batch-1/".
    prefix: Mapped[str] = mapped_column(String(1000), nullable=False)
    # "auto" | "yolo" | "coco" — how the sibling label files are laid out.
    label_format: Mapped[str] = mapped_column(String(20), nullable=False, default="auto")
    dataset_name: Mapped[str | None] = mapped_column(String(200), nullable=True)

    result_dataset_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID, ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True
    )

    total_items: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processed_items: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    celery_task_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    error: Mapped[str | None] = mapped_column(String(2000), nullable=True)
