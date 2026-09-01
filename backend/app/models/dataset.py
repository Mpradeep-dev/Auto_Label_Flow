"""Dataset — a named collection of images/videos within a project.
`DatasetVersion` (immutable pins over the annotation event log) lands in
Phase 5; this file only carries the working container."""
from __future__ import annotations

import uuid
from enum import Enum as PyEnum

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.db.types import GUID, enum_column


class DatasetStatus(str, PyEnum):
    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"


class Dataset(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "datasets"

    project_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[DatasetStatus] = mapped_column(
        enum_column(DatasetStatus, "dataset_status"), nullable=False, default=DatasetStatus.ACTIVE
    )

    project: Mapped["Project"] = relationship(back_populates="datasets")
    images: Mapped[list["Image"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")
    videos: Mapped[list["Video"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")
