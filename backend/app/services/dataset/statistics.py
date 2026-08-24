"""Dataset statistics dashboard (PLAN spec section 13). The auto-label
acceptance rate specifically needs the event log, not just current state:
"accepted" (still AUTO, never touched), "corrected" (AUTO then edited —
source is CORRECTED), and "rejected" (AUTO then deleted) are broken out
separately, matching the spec's worked example (9,800 + 1,850 + 800 =
12,450 — three disjoint buckets over every AUTO prediction ever made, not
just what's currently on screen).
"""
from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.annotation import Annotation, AnnotationEvent, AnnotationEventAction, AnnotationSource
from app.models.image import Image, ImageReviewStatus
from app.models.quality import AnnotationFlag, FlagType


def compute_dataset_statistics(db: Session, dataset_id: uuid.UUID) -> dict:
    total_images = db.scalar(select(func.count()).select_from(Image).where(Image.dataset_id == dataset_id)) or 0
    reviewed_images = (
        db.scalar(
            select(func.count())
            .select_from(Image)
            .where(Image.dataset_id == dataset_id, Image.review_status != ImageReviewStatus.PENDING)
        )
        or 0
    )
    pending_images = total_images - reviewed_images

    image_ids_subq = select(Image.id).where(Image.dataset_id == dataset_id)

    total_annotations = (
        db.scalar(select(func.count()).select_from(Annotation).where(Annotation.image_id.in_(image_ids_subq))) or 0
    )

    per_class = db.execute(
        select(Annotation.class_name, func.count())
        .where(Annotation.image_id.in_(image_ids_subq))
        .group_by(Annotation.class_name)
    ).all()

    per_source = db.execute(
        select(Annotation.source, func.count())
        .where(Annotation.image_id.in_(image_ids_subq))
        .group_by(Annotation.source)
    ).all()
    source_counts = {s.value: c for s, c in per_source}

    avg_confidence = db.scalar(
        select(func.avg(Annotation.confidence)).where(
            Annotation.image_id.in_(image_ids_subq), Annotation.confidence.is_not(None)
        )
    )

    low_confidence_count = (
        db.scalar(
            select(func.count())
            .select_from(AnnotationFlag)
            .where(AnnotationFlag.image_id.in_(image_ids_subq), AnnotationFlag.flag_type == FlagType.LOW_CONFIDENCE)
        )
        or 0
    )
    suspicious_cone_count = (
        db.scalar(
            select(func.count())
            .select_from(AnnotationFlag)
            .where(AnnotationFlag.image_id.in_(image_ids_subq), AnnotationFlag.flag_type == FlagType.SUSPICIOUS_CONE)
        )
        or 0
    )

    # Acceptance-rate breakdown: every annotation_id whose FIRST event was
    # AUTO, bucketed by what happened to it since — see module docstring.
    first_events = (
        select(
            AnnotationEvent.annotation_id,
            func.min(AnnotationEvent.revision_seq).label("first_seq"),
        )
        .where(AnnotationEvent.image_id.in_(image_ids_subq))
        .group_by(AnnotationEvent.annotation_id)
        .subquery()
    )
    auto_originated_ids = db.scalars(
        select(AnnotationEvent.annotation_id)
        .join(
            first_events,
            (AnnotationEvent.annotation_id == first_events.c.annotation_id)
            & (AnnotationEvent.revision_seq == first_events.c.first_seq),
        )
        .where(AnnotationEvent.source == AnnotationSource.AUTO)
    ).all()
    auto_originated_ids = set(auto_originated_ids)
    total_auto_predictions = len(auto_originated_ids)

    if auto_originated_ids:
        live_rows = db.execute(
            select(Annotation.id, Annotation.source).where(Annotation.id.in_(auto_originated_ids))
        ).all()
        live_by_id = {row[0]: row[1] for row in live_rows}
        accepted = sum(1 for aid in auto_originated_ids if live_by_id.get(aid) == AnnotationSource.AUTO)
        corrected = sum(1 for aid in auto_originated_ids if live_by_id.get(aid) == AnnotationSource.CORRECTED)
        rejected = sum(1 for aid in auto_originated_ids if aid not in live_by_id)
    else:
        accepted = corrected = rejected = 0

    acceptance_rate = (accepted / total_auto_predictions) if total_auto_predictions else None

    return {
        "total_images": total_images,
        "reviewed_images": reviewed_images,
        "pending_images": pending_images,
        "completion_pct": round(100 * reviewed_images / total_images, 1) if total_images else 0.0,
        "total_annotations": total_annotations,
        "annotations_by_class": {name: count for name, count in per_class},
        "annotations_by_source": {
            "AUTO": source_counts.get("AUTO", 0),
            "HUMAN": source_counts.get("HUMAN", 0),
            "CORRECTED": source_counts.get("CORRECTED", 0),
        },
        "average_confidence": round(float(avg_confidence), 3) if avg_confidence is not None else None,
        "low_confidence_predictions": low_confidence_count,
        "suspicious_cones": suspicious_cone_count,
        "auto_label_acceptance": {
            "total_auto_predictions": total_auto_predictions,
            "accepted": accepted,
            "corrected": corrected,
            "rejected": rejected,
            "acceptance_rate": round(100 * acceptance_rate, 1) if acceptance_rate is not None else None,
        },
    }
