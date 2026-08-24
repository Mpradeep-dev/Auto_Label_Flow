"""Computes and persists `ImagePoseContext` for one image — the one place
pose_v1 (or whatever auxiliary pose model a project configures) runs.
Called from wherever detection already runs (single-image auto-annotate,
batch inference task) whenever `project.pose_model_id` is set, so pose
context is always computed alongside detection, never as an afterthought
— see PLAN "This is why pose_v1 runs alongside detect_v1 on this project
even though person is never itself an annotation class."
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np
from sqlalchemy.orm import Session

from app.models.quality import ImagePoseContext
from app.services.inference.registry import get_pose_model
from app.services.quality.body_scale import compute_body_scale


def compute_and_store_pose_context(
    db: Session, *, image_id: uuid.UUID, pose_model_id: uuid.UUID, image_bgr: np.ndarray, aspect: float
) -> ImagePoseContext:
    pose_model = get_pose_model(db, pose_model_id)
    people = pose_model.predict(image_bgr)

    persons_json = []
    for person in people:
        scale, source = compute_body_scale(person.keypoints, aspect=aspect)
        persons_json.append(
            {
                "x1": person.x1,
                "y1": person.y1,
                "x2": person.x2,
                "y2": person.y2,
                "confidence": person.confidence,
                "keypoints": [{"x": k.x, "y": k.y, "confidence": k.confidence} for k in person.keypoints],
                "body_scale": scale,
                "body_scale_source": source,
            }
        )

    existing = db.query(ImagePoseContext).filter(ImagePoseContext.image_id == image_id).one_or_none()
    if existing is not None:
        existing.persons = persons_json
        existing.model_id = pose_model_id
        existing.computed_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(existing)
        return existing

    context = ImagePoseContext(
        image_id=image_id, model_id=pose_model_id, persons=persons_json, computed_at=datetime.now(timezone.utc)
    )
    db.add(context)
    db.commit()
    db.refresh(context)
    return context
