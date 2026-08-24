"""Error analysis (PLAN spec section 15): aggregates the error_category /
error_reason a reviewer recorded when deleting a prediction. Scoped to
what the annotation service actually captures today — DELETE events with
an error_category (see `services/annotation/service.py::delete_annotation`).
A class correction via PUT (AUTO -> CORRECTED) doesn't currently carry its
own error taxonomy the way a delete does; that's a real gap, not something
papered over here — this view only reports what reviewers explicitly
categorized.
"""
from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.annotation import AnnotationEvent, AnnotationEventAction
from app.models.image import Image


def compute_error_analysis(db: Session, dataset_id: uuid.UUID) -> dict:
    image_ids_subq = select(Image.id).where(Image.dataset_id == dataset_id)

    by_category = db.execute(
        select(AnnotationEvent.error_category, func.count())
        .where(
            AnnotationEvent.image_id.in_(image_ids_subq),
            AnnotationEvent.action == AnnotationEventAction.DELETE,
            AnnotationEvent.error_category.is_not(None),
        )
        .group_by(AnnotationEvent.error_category)
    ).all()

    by_reason = db.execute(
        select(AnnotationEvent.error_reason, func.count())
        .where(
            AnnotationEvent.image_id.in_(image_ids_subq),
            AnnotationEvent.action == AnnotationEventAction.DELETE,
            AnnotationEvent.error_reason.is_not(None),
        )
        .group_by(AnnotationEvent.error_reason)
    ).all()

    total_categorized = sum(count for _, count in by_category)

    return {
        "total_categorized_deletions": total_categorized,
        "by_category": {cat.value: count for cat, count in by_category},
        "by_reason": {reason.value: count for reason, count in by_reason},
    }
