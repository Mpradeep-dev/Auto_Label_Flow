"""Annotation storage: an insert-only event log plus a maintained
current-state projection (PLAN "Core design decision 2").

`annotation_events` is the audit trail — one row per edit, never updated or
deleted. `annotations` is a rebuildable cache of "where does this box stand
right now", upserted in the same transaction as the event it came from, so
the review loop's hottest query (`WHERE image_id = ?`) stays a single
indexed read instead of an event replay. Dataset versioning (Phase 5) pins
`(annotation_id, event_id)` pairs into a separate table rather than copying
rows — see `services/annotation/` for the transaction that keeps both
tables consistent; nothing should INSERT/UPDATE `annotations` directly
outside that service.
"""
from __future__ import annotations

import uuid
from enum import Enum as PyEnum

from sqlalchemy import Enum, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class AnnotationSource(str, PyEnum):
    AUTO = "AUTO"
    HUMAN = "HUMAN"
    CORRECTED = "CORRECTED"


class AnnotationReviewStatus(str, PyEnum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class AnnotationEventAction(str, PyEnum):
    CREATE = "CREATE"
    UPDATE = "UPDATE"
    DELETE = "DELETE"


class ErrorCategory(str, PyEnum):
    """PLAN spec section 15 — recorded when a human deletes or reclassifies
    a prediction, so the reason is queryable later for targeted model
    improvement, not just "it was wrong"."""

    FALSE_POSITIVE = "FALSE_POSITIVE"
    FALSE_NEGATIVE = "FALSE_NEGATIVE"
    WRONG_CLASS = "WRONG_CLASS"
    BAD_BBOX = "BAD_BBOX"
    DUPLICATE = "DUPLICATE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    OCCLUSION = "OCCLUSION"
    BLUR = "BLUR"
    SMALL_OBJECT = "SMALL_OBJECT"


class ErrorReason(str, PyEnum):
    """Sub-reason for FALSE_POSITIVE specifically — the spec's concrete
    example: 'a human deletes a cone because it's actually a player's
    foot'. Optional on every event; only meaningful alongside
    FALSE_POSITIVE."""

    PLAYER_FOOT = "PLAYER_FOOT"
    PLAYER_SHOE = "PLAYER_SHOE"
    SHADOW = "SHADOW"
    FOOTBALL = "FOOTBALL"
    GRASS_BACKGROUND = "GRASS_BACKGROUND"
    OTHER = "OTHER"


class AnnotationEvent(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Insert-only. `annotation_id` is stable across every revision of one
    box — NOT the same as this row's own `id`, which identifies the event
    itself. `revision_seq` orders events per annotation_id for replay."""

    __tablename__ = "annotation_events"

    annotation_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False, index=True)
    revision_seq: Mapped[int] = mapped_column(Integer, nullable=False)
    image_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("images.id", ondelete="CASCADE"), nullable=False, index=True
    )

    action: Mapped[AnnotationEventAction] = mapped_column(
        Enum(AnnotationEventAction, name="annotation_event_action"), nullable=False
    )

    class_id: Mapped[int] = mapped_column(Integer, nullable=False)
    class_name: Mapped[str] = mapped_column(String(100), nullable=False)
    x1: Mapped[float] = mapped_column(Float, nullable=False)
    y1: Mapped[float] = mapped_column(Float, nullable=False)
    x2: Mapped[float] = mapped_column(Float, nullable=False)
    y2: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[AnnotationSource] = mapped_column(Enum(AnnotationSource, name="annotation_source"), nullable=False)

    error_category: Mapped[ErrorCategory | None] = mapped_column(
        Enum(ErrorCategory, name="error_category"), nullable=True
    )
    error_reason: Mapped[ErrorReason | None] = mapped_column(Enum(ErrorReason, name="error_reason"), nullable=True)

    actor: Mapped[str | None] = mapped_column(String(200), nullable=True)


class Annotation(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """The live projection — current state of one annotation. `id` here
    equals `AnnotationEvent.annotation_id` for every event in that box's
    history (enforced by the service layer, not the DB, since the id is
    assigned before the first event is written)."""

    __tablename__ = "annotations"

    image_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("images.id", ondelete="CASCADE"), nullable=False, index=True
    )
    class_id: Mapped[int] = mapped_column(Integer, nullable=False)
    class_name: Mapped[str] = mapped_column(String(100), nullable=False)
    x1: Mapped[float] = mapped_column(Float, nullable=False)
    y1: Mapped[float] = mapped_column(Float, nullable=False)
    x2: Mapped[float] = mapped_column(Float, nullable=False)
    y2: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[AnnotationSource] = mapped_column(Enum(AnnotationSource, name="annotation_source"), nullable=False)
    review_status: Mapped[AnnotationReviewStatus] = mapped_column(
        Enum(AnnotationReviewStatus, name="annotation_review_status"),
        nullable=False,
        default=AnnotationReviewStatus.PENDING,
    )
    latest_event_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    revision_seq: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
