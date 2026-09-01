"""Quality flags + pose context (PLAN Phase 7 / Core design decision 1).

`AnnotationFlag` rows are derived data, not part of the immutable event log
— recomputed and replaced whenever the annotation they reference changes,
cascade-deleted with it. Only TRIGGERED flags are persisted; a rule that
abstains (e.g. no person detected) or doesn't fire simply writes no row.

`ImagePoseContext` is the one place pose_v1's output lives — never as
annotations (PLAN Decision 1: person is never an annotation class), purely
as context for pose-dependent quality rules.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import Float, ForeignKey, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.db.types import GUID, TZDateTime, enum_column


class FlagType(str, PyEnum):
    CONE_NEAR_PLAYER = "CONE_NEAR_PLAYER"
    SUSPICIOUS_CONE = "SUSPICIOUS_CONE"
    VERY_SMALL_CONE = "VERY_SMALL_CONE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    POSSIBLE_DUPLICATE = "POSSIBLE_DUPLICATE"
    ISOLATED_DETECTION = "ISOLATED_DETECTION"
    TEMPORAL_ANOMALY = "TEMPORAL_ANOMALY"


class FlagResolution(str, PyEnum):
    CONFIRMED_FP = "CONFIRMED_FP"  # reviewer agrees this was a false positive
    CONFIRMED_OK = "CONFIRMED_OK"  # reviewer looked and it's fine, flag dismissed
    EDITED = "EDITED"  # reviewer corrected the box, which resolves the flag


class AnnotationFlag(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "annotation_flags"

    annotation_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("annotations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    image_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("images.id", ondelete="CASCADE"), nullable=False, index=True
    )
    flag_type: Mapped[FlagType] = mapped_column(enum_column(FlagType, "flag_type"), nullable=False, index=True)
    severity: Mapped[float] = mapped_column(Float, nullable=False)  # 0-1, higher = more suspicious
    reason: Mapped[str] = mapped_column(String(500), nullable=False)  # human-readable, shown in the UI
    details: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # numeric evidence, e.g. {"distance_bl": 0.11}
    rule_version: Mapped[str] = mapped_column(String(50), nullable=False, default="v1")

    resolution: Mapped[FlagResolution | None] = mapped_column(enum_column(FlagResolution, "flag_resolution"), nullable=True)
    resolved_by: Mapped[str | None] = mapped_column(String(200), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(TZDateTime, nullable=True)


class ImagePoseContext(UUIDPrimaryKeyMixin, Base):
    """One row per image (whether standalone or a video frame) — computed
    once, alongside detection, by whichever inference path touched that
    image (see services/inference/pose_context.py)."""

    __tablename__ = "image_pose_context"

    image_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("images.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    model_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID, ForeignKey("models.id", ondelete="SET NULL"), nullable=True
    )
    # [{"x1","y1","x2","y2","confidence","keypoints":[{"x","y","confidence"}]*17,
    #   "body_scale","body_scale_source"}, ...] — one entry per detected person.
    persons: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    computed_at: Mapped[datetime] = mapped_column(TZDateTime, nullable=False)
