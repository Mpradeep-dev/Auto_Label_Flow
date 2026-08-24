"""Orchestrates one image's quality pass: build FrameContext from stored
pose context + annotations (+ optional precomputed track stats for video
frames), run PredictionQualityAnalyzer, replace that image's flags, and
recompute its difficulty score — the single call site both the synchronous
per-image path and the batch Celery task use, so they can never drift.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.image import Image
from app.models.project import Project
from app.models.quality import AnnotationFlag, FlagType, ImagePoseContext
from app.services.annotation.service import list_annotations_for_image
from app.services.inference.detector import Keypoint
from app.services.quality.analyzer import PredictionQualityAnalyzer
from app.services.quality.context import AnnotationLike, FrameContext, PersonContext, TrackStats
from app.services.review.difficulty import DifficultyInputs, compute_difficulty


def _build_frame_context(
    db: Session,
    image: Image,
    is_video_frame: bool,
    track_stats_by_annotation_id: dict[uuid.UUID, TrackStats],
) -> tuple[FrameContext, list[AnnotationLike]]:
    pose_ctx = db.scalar(select(ImagePoseContext).where(ImagePoseContext.image_id == image.id))
    persons: list[PersonContext] = []
    if pose_ctx is not None:
        for p in pose_ctx.persons:
            keypoints = [Keypoint(x=k["x"], y=k["y"], confidence=k["confidence"]) for k in p["keypoints"]]
            persons.append(
                PersonContext(
                    x1=p["x1"],
                    y1=p["y1"],
                    x2=p["x2"],
                    y2=p["y2"],
                    keypoints=keypoints,
                    body_scale=p["body_scale"],
                    body_scale_source=p["body_scale_source"],
                )
            )

    annotations = list_annotations_for_image(db, image.id)
    ann_like = [
        AnnotationLike(
            id=str(a.id), class_id=a.class_id, class_name=a.class_name, confidence=a.confidence,
            x1=a.x1, y1=a.y1, x2=a.x2, y2=a.y2,
        )
        for a in annotations
    ]
    aspect = image.width / image.height if image.height else 1.0
    track_stats_str_keyed = {str(k): v for k, v in track_stats_by_annotation_id.items()}

    context = FrameContext(
        image_id=str(image.id),
        aspect=aspect,
        all_annotations=ann_like,
        persons=persons,
        pose_available=pose_ctx is not None,
        track_stats_by_annotation_id=track_stats_str_keyed,
        is_video_frame=is_video_frame,
    )
    return context, ann_like


def analyze_image_quality(
    db: Session,
    *,
    image: Image,
    project: Project,
    is_video_frame: bool = False,
    track_stats_by_annotation_id: dict[uuid.UUID, TrackStats] | None = None,
) -> list[AnnotationFlag]:
    context, annotations = _build_frame_context(db, image, is_video_frame, track_stats_by_annotation_id or {})

    triggered = PredictionQualityAnalyzer().analyze(annotations, context, project)

    db.query(AnnotationFlag).filter(AnnotationFlag.image_id == image.id).delete()
    flag_rows = [
        AnnotationFlag(
            annotation_id=uuid.UUID(t.annotation_id),
            image_id=image.id,
            flag_type=FlagType(t.flag_type),
            severity=t.severity,
            reason=t.reason,
            details=t.details,
        )
        for t in triggered
    ]
    db.add_all(flag_rows)

    has_temporal_anomaly = any(t.flag_type == FlagType.TEMPORAL_ANOMALY.value for t in triggered)
    difficulty_inputs = DifficultyInputs(
        flag_severities=[t.severity for t in triggered],
        confidences=[a.confidence for a in annotations if a.confidence is not None],
        has_temporal_anomaly=has_temporal_anomaly,
        pose_unavailable=is_video_frame and project.pose_model_id is not None and not context.pose_available,
    )
    score, components = compute_difficulty(difficulty_inputs)
    image.difficulty_score = score
    image.difficulty_components = components

    db.commit()
    for row in flag_rows:
        db.refresh(row)
    return flag_rows
