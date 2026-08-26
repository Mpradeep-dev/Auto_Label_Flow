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
    ShapeType,
)
from app.models.image import Image, ImageReviewStatus
from app.models.project import Project


class AnnotationNotFoundError(LookupError):
    pass


class InvalidGeometryError(ValueError):
    """Raised for a POLYGON with too few points. The API layer catches this
    and returns 400, the same way it already handles the x2<=x1 bbox check."""


def _bbox_from_points(points: list[list[float]]) -> tuple[float, float, float, float]:
    """The single point of truth for a polygon's bounding box: the server
    always recomputes x1..y2 from `points` on every create/update, never
    trusts a client-supplied bbox for a POLYGON row — this is what keeps the
    bbox columns authoritative for bbox-approximate downstream consumers
    (quality rules, SORT tracker, exporters)."""
    if len(points) < 3:
        raise InvalidGeometryError("A polygon needs at least 3 points")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _translate_points(points: list[list[float]], dx: float, dy: float) -> list[list[float]]:
    return [[min(1.0, p[0] + dx), min(1.0, p[1] + dy)] for p in points]


def _write_event(
    db: Session,
    *,
    annotation_id: uuid.UUID,
    revision_seq: int,
    image_id: uuid.UUID,
    action: AnnotationEventAction,
    class_id: int,
    class_name: str,
    shape_type: ShapeType,
    points: list[list[float]] | None,
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
        shape_type=shape_type,
        points=points,
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
    confidence: float | None,
    source: AnnotationSource,
    shape_type: ShapeType = ShapeType.BBOX,
    points: list[list[float]] | None = None,
    x1: float | None = None,
    y1: float | None = None,
    x2: float | None = None,
    y2: float | None = None,
    actor: str | None = None,
) -> Annotation:
    """Used both for a single HUMAN-drawn shape (Phase 3 "Add annotation")
    and, in bulk, for AUTO predictions from a model run (see
    `bulk_create_from_predictions`). For shape_type=POLYGON, x1..y2 are
    ignored if given and always recomputed from `points` (see
    `_bbox_from_points`) — for BBOX, x1..y2 are required and used as-is."""
    if shape_type == ShapeType.POLYGON:
        x1, y1, x2, y2 = _bbox_from_points(points or [])
    elif x1 is None or y1 is None or x2 is None or y2 is None:
        raise InvalidGeometryError("x1, y1, x2, y2 are required for a BBOX annotation")

    annotation_id = uuid.uuid4()
    event = _write_event(
        db,
        annotation_id=annotation_id,
        revision_seq=1,
        image_id=image_id,
        action=AnnotationEventAction.CREATE,
        class_id=class_id,
        class_name=class_name,
        shape_type=shape_type,
        points=points,
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
        shape_type=shape_type,
        points=points,
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


def _resolve_class_id(project: Project, class_name: str) -> int:
    """Class identity is resolved by NAME against the project's
    `class_config` — same rule the Roboflow/COCO/CVAT importers use
    (PLAN "class taxonomy is read from the model, never hardcoded"): a raw
    id is only meaningful within whatever produced it (there, an external
    export; here, one detector's own weights), never assumed to match this
    project's own ids. A name already in class_config (matched
    case/whitespace-insensitively, same as the frontend's "add class"
    dedupe) reuses its id; a name the project has never seen is appended
    under a fresh project-local id — otherwise a model that detects a class
    outside the project's existing taxonomy (e.g. a "cone" class the
    project was never told about) would produce AUTO boxes for a class the
    draw-picker can never offer, since the picker only lists class_config."""
    existing = list(project.class_config or [])
    name_lower = class_name.strip().lower()
    for entry in existing:
        if entry["name"].strip().lower() == name_lower:
            return entry["id"]
    next_id = (max((entry["id"] for entry in existing), default=-1)) + 1
    existing.append({"id": next_id, "name": class_name})
    project.class_config = existing
    return next_id


def bulk_create_from_predictions(
    db: Session,
    *,
    image_id: uuid.UUID,
    project_id: uuid.UUID,
    predictions: list[dict],
) -> list[Annotation]:
    """`predictions` items: {class_id, class_name, confidence, x1, y1, x2, y2}
    for a BBOX prediction, or {..., shape_type: "POLYGON", points: [[x,y],...]}
    for a polygon/SAM-derived one (x1..y2 then optional — recomputed from
    points, see `create_annotation`). Always source=AUTO — this is the one
    place model output enters the annotation table (PLAN section 27: "Every
    auto-generated annotation must retain source=AUTO until a human approves
    or modifies it").

    `class_id` on each prediction is the detector's own index and gets
    discarded in favor of `_resolve_class_id`, which maps `class_name` onto
    (and if needed, extends) the project's own class_config."""
    project = db.get(Project, project_id)
    created: list[Annotation] = []
    for pred in predictions:
        class_id = _resolve_class_id(project, pred["class_name"]) if project is not None else pred["class_id"]
        created.append(
            create_annotation(
                db,
                image_id=image_id,
                class_id=class_id,
                class_name=pred["class_name"],
                shape_type=ShapeType(pred.get("shape_type", ShapeType.BBOX)),
                points=pred.get("points"),
                x1=pred.get("x1"),
                y1=pred.get("y1"),
                x2=pred.get("x2"),
                y2=pred.get("y2"),
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
    points: list[list[float]] | None = None,
    confidence: float | None = None,
    actor: str | None = None,
) -> Annotation:
    """Moving/resizing a shape (box corners, or polygon points) or changing
    its class all count as a correction: AUTO -> CORRECTED. A HUMAN-drawn
    shape stays HUMAN (there is no "more human" state). CORRECTED stays
    CORRECTED. This is the service-layer rule PLAN Decision 2 requires —
    callers never set `source` directly on an edit.

    `shape_type` is immutable after creation (no bbox<->polygon conversion
    in place — delete and redraw as the other type) — this function
    deliberately has no `shape_type` parameter. `points` is only meaningful
    for an existing POLYGON annotation; `x1..y2` are only meaningful for an
    existing BBOX one, since a POLYGON's bbox is always derived from its
    points, never edited directly."""
    annotation = db.get(Annotation, annotation_id)
    if annotation is None:
        raise AnnotationNotFoundError(str(annotation_id))

    if annotation.shape_type == ShapeType.POLYGON:
        if x1 is not None or y1 is not None or x2 is not None or y2 is not None:
            raise InvalidGeometryError("x1..y2 cannot be edited directly on a POLYGON annotation; edit `points`")
        new_points = points if points is not None else annotation.points
        new_x1, new_y1, new_x2, new_y2 = _bbox_from_points(new_points)
    else:
        if points is not None:
            raise InvalidGeometryError("`points` cannot be set on a BBOX annotation")
        new_points = None
        new_x1 = x1 if x1 is not None else annotation.x1
        new_y1 = y1 if y1 is not None else annotation.y1
        new_x2 = x2 if x2 is not None else annotation.x2
        new_y2 = y2 if y2 is not None else annotation.y2

    new_class_id = class_id if class_id is not None else annotation.class_id
    new_class_name = class_name if class_name is not None else annotation.class_name
    new_confidence = confidence if confidence is not None else annotation.confidence

    geometry_or_class_changed = (
        new_class_id != annotation.class_id
        or new_x1 != annotation.x1
        or new_y1 != annotation.y1
        or new_x2 != annotation.x2
        or new_y2 != annotation.y2
        or new_points != annotation.points
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
        shape_type=annotation.shape_type,
        points=new_points,
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
    annotation.points = new_points
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
        shape_type=annotation.shape_type,
        points=annotation.points,
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
    HUMAN-sourced (it's a new shape a person just created), offset slightly
    so it doesn't render exactly on top of the original — for a polygon,
    every vertex is offset by the same amount rather than touching x1..y2
    directly, which would desync points from their derived bbox."""
    original = db.get(Annotation, annotation_id)
    if original is None:
        raise AnnotationNotFoundError(str(annotation_id))

    offset = 0.02
    if original.shape_type == ShapeType.POLYGON:
        return create_annotation(
            db,
            image_id=original.image_id,
            class_id=original.class_id,
            class_name=original.class_name,
            shape_type=ShapeType.POLYGON,
            points=_translate_points(original.points, offset, offset),
            confidence=None,
            source=AnnotationSource.HUMAN,
            actor=actor,
        )
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
