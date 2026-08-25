"""Dataset versioning — immutable pins over the annotation event log
(PLAN "Append-only event log + projection + version pins"), not
copy-on-create row duplication. Creating a version writes only
`(annotation_id -> event_id)` pointers; export reads the pinned events, so
an old version stays byte-reproducible no matter how much the live
`annotations` table moves afterward.
"""
from __future__ import annotations

import uuid
from enum import Enum as PyEnum

from sqlalchemy import Enum, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class DatasetVersionStatus(str, PyEnum):
    DRAFT = "DRAFT"
    EXPORTING = "EXPORTING"
    EXPORTED = "EXPORTED"
    FAILED = "FAILED"


class SplitName(str, PyEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class DatasetVersion(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "dataset_versions"
    __table_args__ = (UniqueConstraint("dataset_id", "version_number", name="uq_dataset_version_number"),)

    dataset_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[DatasetVersionStatus] = mapped_column(
        Enum(DatasetVersionStatus, name="dataset_version_status"), nullable=False, default=DatasetVersionStatus.DRAFT
    )

    train_ratio: Mapped[float] = mapped_column(Float, nullable=False, default=0.8)
    val_ratio: Mapped[float] = mapped_column(Float, nullable=False, default=0.1)
    test_ratio: Mapped[float] = mapped_column(Float, nullable=False, default=0.1)
    split_seed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    used_frame_level_fallback: Mapped[bool] = mapped_column(nullable=False, default=False)

    total_images: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_annotations: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    export_storage_key: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    # Same YOLO zip this version always had, plus two more formats a version
    # can independently be exported to — each is generated on demand and
    # cached here so a re-download doesn't regenerate it. COCO/CVAT-XML are
    # the formats CVAT itself imports/exports, so this is also the bridge
    # into/out of CVAT without a live API integration (see
    # `services/dataset/export_coco.py` / `export_cvat.py`).
    coco_export_storage_key: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    cvat_export_storage_key: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    error: Mapped[str | None] = mapped_column(String(2000), nullable=True)


class DatasetVersionImage(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One image's split assignment within one version. `image_id` is not a
    hard FK (ON DELETE would silently corrupt a supposedly-immutable
    version) — deleting an image after it's pinned into a version is an
    edge case handled at export time (skip + warn), not by cascading."""

    __tablename__ = "dataset_version_images"
    __table_args__ = (UniqueConstraint("dataset_version_id", "image_id", name="uq_version_image"),)

    dataset_version_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    image_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False, index=True)
    split: Mapped[SplitName] = mapped_column(Enum(SplitName, name="split_name"), nullable=False)
    source_group_id: Mapped[str] = mapped_column(String(100), nullable=False)


class DatasetVersionAnnotationPin(UUIDPrimaryKeyMixin, Base):
    """The actual pin: freezes one annotation at one historical event. Export
    reads `AnnotationEvent` rows by `pinned_event_id`, never the live
    `annotations` table — that's what makes a version reproducible after
    the live data keeps moving."""

    __tablename__ = "dataset_version_annotation_pins"

    dataset_version_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    annotation_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False, index=True)
    pinned_event_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    image_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False, index=True)
