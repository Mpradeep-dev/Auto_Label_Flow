"""The only code allowed to write to `annotations` or `annotation_events`
(PLAN Core design decision 2) — every mutation goes through here so the
event log and the projection can never drift apart, and so the
AUTO -> CORRECTED source transition rule lives in exactly one place.
"""
from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.annotation import (
    Annotation,
    AnnotationEvent,
    AnnotationEventAction,
    AnnotationReviewStatus,
    AnnotationSource,
    ErrorCategory,
    ErrorReason,
)
from app.models.image import Image, ImageReviewStatus


class AnnotationNotFoundError(LookupError):
    pass


def _write_event(
    db: Session,
    *,
    annotation_id: uuid.UUID,
    revision_seq: int,
    image_id: uuid.UUID,
    action: AnnotationEventAction,
    class_id: int,
    class_name: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    confidence: float | None,
    source: AnnotationSource,
    actor: str | None,
    error_category: ErrorCategory | None = None,
    error_reason: ErrorReason | None = None,
) -> AnnotationEvent:
    event = AnnotationEvent(
        annotation_id=annotation_id,
        revision_seq=revision_seq,
        image_id=image_id,
        action=action,
        class_id=class_id,
        class_name=class_name,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        confidence=confidence,
        source=source,
        error_category=error_category,
        error_reason=error_reason,
        actor=actor,
    )
    db.add(event)
    db.flush()  # need event.id before writing it onto the projection row
    return event


def create_annotation(
    db: Session,
    *,
    image_id: uuid.UUID,
    class_id: int,
    class_name: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    confidence: float | None,
    source: AnnotationSource,
    actor: str | None = None,
) -> Annotation:
    """Used both for a single HUMAN-drawn box (Phase 3 "Add annotation")
    and, in bulk, for AUTO predictions from a model run (see
    `bulk_create_from_predictions`)."""
    annotation_id = uuid.uuid4()
    event = _write_event(
        db,
        annotation_id=annotation_id,
        revision_seq=1,
        image_id=image_id,
        action=AnnotationEventAction.CREATE,
        class_id=class_id,
        class_name=class_name,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        confidence=confidence,
        source=source,
        actor=actor,
    )
    annotation = Annotation(
        id=annotation_id,
        image_id=image_id,
        class_id=class_id,
        class_name=class_name,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        confidence=confidence,
        source=source,
        review_status=AnnotationReviewStatus.PENDING,
        latest_event_id=event.id,
        revision_seq=1,
    )
    db.add(annotation)
    db.commit()
    db.refresh(annotation)
    return annotation


def bulk_create_from_predictions(
    db: Session,
    *,
    image_id: uuid.UUID,
    predictions: list[dict],
) -> list[Annotation]:
    """`predictions` items: {class_id, class_name, confidence, x1, y1, x2, y2}.
    Always source=AUTO — this is the one place model output enters the
    annotation table (PLAN section 27: "Every auto-generated annotation
    must retain source=AUTO until a human approves or modifies it")."""
    created: list[Annotation] = []
    for pred in predictions:
        created.append(
            create_annotation(
                db,
                image_id=image_id,
                class_id=pred["class_id"],
                class_name=pred["class_name"],
                x1=pred["x1"],
                y1=pred["y1"],
                x2=pred["x2"],
                y2=pred["y2"],
                confidence=pred["confidence"],
                source=AnnotationSource.AUTO,
                actor=None,
            )
        )
    return created


def update_annotation(
    db: Session,
    *,
    annotation_id: uuid.UUID,
    class_id: int | None = None,
    class_name: str | None = None,
    x1: float | None = None,
    y1: float | None = None,
    x2: float | None = None,
    y2: float | None = None,
    confidence: float | None = None,
    actor: str | None = None,
) -> Annotation:
    """Moving a box, resizing it, or changing its class all count as a
    correction: AUTO -> CORRECTED. A HUMAN-drawn box stays HUMAN (there is
    no "more human" state). CORRECTED stays CORRECTED. This is the
    service-layer rule PLAN Decision 2 requires — callers never set
    `source` directly on an edit."""
    annotation = db.get(Annotation, annotation_id)
    if annotation is None:
        raise AnnotationNotFoundError(str(annotation_id))

    new_class_id = class_id if class_id is not None else annotation.class_id
    new_class_name = class_name if class_name is not None else annotation.class_name
    new_x1 = x1 if x1 is not None else annotation.x1
    new_y1 = y1 if y1 is not None else annotation.y1
    new_x2 = x2 if x2 is not None else annotation.x2
    new_y2 = y2 if y2 is not None else annotation.y2
    new_confidence = confidence if confidence is not None else annotation.confidence

    geometry_or_class_changed = (
        new_class_id != annotation.class_id
        or new_x1 != annotation.x1
        or new_y1 != annotation.y1
        or new_x2 != annotation.x2
        or new_y2 != annotation.y2
    )
    new_source = annotation.source
    if geometry_or_class_changed and annotation.source == AnnotationSource.AUTO:
        new_source = AnnotationSource.CORRECTED

    next_seq = annotation.revision_seq + 1
    event = _write_event(
        db,
        annotation_id=annotation.id,
        revision_seq=next_seq,
        image_id=annotation.image_id,
        action=AnnotationEventAction.UPDATE,
        class_id=new_class_id,
        class_name=new_class_name,
        x1=new_x1,
        y1=new_y1,
        x2=new_x2,
        y2=new_y2,
        confidence=new_confidence,
        source=new_source,
        actor=actor,
    )

    annotation.class_id = new_class_id
    annotation.class_name = new_class_name
    annotation.x1 = new_x1
    annotation.y1 = new_y1
    annotation.x2 = new_x2
    annotation.y2 = new_y2
    annotation.confidence = new_confidence
    annotation.source = new_source
    annotation.latest_event_id = event.id
    annotation.revision_seq = next_seq

    db.commit()
    db.refresh(annotation)
    return annotation


def delete_annotation(
    db: Session,
    *,
    annotation_id: uuid.UUID,
    actor: str | None = None,
    error_category: ErrorCategory | None = None,
    error_reason: ErrorReason | None = None,
) -> None:
    """Removes the row from the live projection but writes a DELETE event
    first — the box's full history (including this deletion, with its
    error category/reason for the Phase 8 error-analysis view) survives in
    `annotation_events` forever."""
    annotation = db.get(Annotation, annotation_id)
    if annotation is None:
        raise AnnotationNotFoundError(str(annotation_id))

    _write_event(
        db,
        annotation_id=annotation.id,
        revision_seq=annotation.revision_seq + 1,
        image_id=annotation.image_id,
        action=AnnotationEventAction.DELETE,
        class_id=annotation.class_id,
        class_name=annotation.class_name,
        x1=annotation.x1,
        y1=annotation.y1,
        x2=annotation.x2,
        y2=annotation.y2,
        confidence=annotation.confidence,
        source=annotation.source,
        actor=actor,
        error_category=error_category,
        error_reason=error_reason,
    )
    db.delete(annotation)
    db.commit()


def duplicate_annotation(db: Session, *, annotation_id: uuid.UUID, actor: str | None = None) -> Annotation:
    """PLAN spec section 2: right-panel "Duplicate" action. The copy is
    HUMAN-sourced (it's a new box a person just created), offset slightly
    so it doesn't render exactly on top of the original."""
    original = db.get(Annotation, annotation_id)
    if original is None:
        raise AnnotationNotFoundError(str(annotation_id))

    offset = 0.02
    return create_annotation(
        db,
        image_id=original.image_id,
        class_id=original.class_id,
        class_name=original.class_name,
        x1=min(1.0, original.x1 + offset),
        y1=min(1.0, original.y1 + offset),
        x2=min(1.0, original.x2 + offset),
        y2=min(1.0, original.y2 + offset),
        confidence=None,
        source=AnnotationSource.HUMAN,
        actor=actor,
    )


def list_annotations_for_image(db: Session, image_id: uuid.UUID) -> list[Annotation]:
    return list(db.scalars(select(Annotation).where(Annotation.image_id == image_id).order_by(Annotation.created_at)))


def approve_image(db: Session, *, image_id: uuid.UUID) -> Image:
    """PLAN spec section 3: 'Approve' marks the image reviewed WITHOUT
    touching annotation source — an AUTO box that was never edited stays
    AUTO; only its review_status moves to APPROVED. This is exactly the
    distinction the acceptance-rate metric (Phase 8) depends on."""
    image = db.get(Image, image_id)
    if image is None:
        raise AnnotationNotFoundError(str(image_id))

    for annotation in list_annotations_for_image(db, image_id):
        annotation.review_status = AnnotationReviewStatus.APPROVED
    image.review_status = ImageReviewStatus.APPROVED
    db.commit()
    db.refresh(image)
    return image


def reject_image(db: Session, *, image_id: uuid.UUID) -> Image:
    image = db.get(Image, image_id)
    if image is None:
        raise AnnotationNotFoundError(str(image_id))

    for annotation in list_annotations_for_image(db, image_id):
        annotation.review_status = AnnotationReviewStatus.REJECTED
    image.review_status = ImageReviewStatus.REJECTED
    db.commit()
    db.refresh(image)
    return image
