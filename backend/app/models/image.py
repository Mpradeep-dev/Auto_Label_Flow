"""Image — a single annotatable frame, whether directly uploaded or
extracted from a Video. `source_type`/`video_id`/`frame_index`/
`frame_timestamp_s` are what make a video frame traceable back to its
source video for debugging (PLAN section 7)."""
from __future__ import annotations

import uuid
from enum import Enum as PyEnum

from sqlalchemy import JSON, Enum, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class ImageSourceType(str, PyEnum):
    UPLOAD = "UPLOAD"
    VIDEO_FRAME = "VIDEO_FRAME"


class ImageReviewStatus(str, PyEnum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class Image(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "images"

    project_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )

    storage_key: Mapped[str] = mapped_column(String(1000), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)

    source_type: Mapped[ImageSourceType] = mapped_column(
        Enum(ImageSourceType, name="image_source_type"), nullable=False, default=ImageSourceType.UPLOAD
    )
    video_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=True, index=True
    )
    frame_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frame_timestamp_s: Mapped[float | None] = mapped_column(Float, nullable=True)

    review_status: Mapped[ImageReviewStatus] = mapped_column(
        Enum(ImageReviewStatus, name="image_review_status"),
        nullable=False,
        default=ImageReviewStatus.PENDING,
        index=True,
    )

    # Populated by the Phase 7 quality/active-learning pass; NULL until then
    # (queue ordering falls back to created_at when NULL — see review/queue.py).
    difficulty_score: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    # Raw sub-signals behind difficulty_score (flag severity, confidence,
    # temporal anomaly, pose availability, prior corrections), stored
    # separately so reweighting the formula later is a cheap recombination
    # of stored components, not a full geometry/pose re-run.
    difficulty_components: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    dataset: Mapped["Dataset"] = relationship(back_populates="images")
    video: Mapped["Video | None"] = relationship(back_populates="frames")
