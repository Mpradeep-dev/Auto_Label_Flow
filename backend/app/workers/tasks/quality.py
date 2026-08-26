"""Batch quality analysis — runs on the `gpu` queue because it may need to
run the pose model (real inference), not because the rule evaluation
itself is GPU-bound. Groups video frames by video_id and runs the temporal
pass (SortTracker) once per video before analyzing its frames, so
ISOLATED_DETECTION/TEMPORAL_ANOMALY have track data; standalone images
skip straight to per-image analysis."""
from __future__ import annotations

import logging
import uuid

import cv2
import numpy as np
from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.image import Image
from app.models.project import Project
from app.models.quality import ImagePoseContext
from app.services.inference.pose_context import compute_and_store_pose_context
from app.services.quality.run_analysis import analyze_image_quality
from app.services.quality.temporal_pass import run_temporal_pass_for_video
from app.services.storage.factory import get_storage
from app.workers.celery_app import QUALITY_SOFT_TIME_LIMIT_S, QUALITY_TIME_LIMIT_S, celery_app

logger = logging.getLogger(__name__)


def _ensure_pose_context(db, image: Image, project: Project) -> None:
    if project.pose_model_id is None:
        return
    existing = db.scalar(select(ImagePoseContext).where(ImagePoseContext.image_id == image.id))
    if existing is not None:
        return
    data = get_storage().read_bytes(image.storage_key)
    arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return
    aspect = image.width / image.height if image.height else 1.0
    compute_and_store_pose_context(
        db, image_id=image.id, pose_model_id=project.pose_model_id, image_bgr=arr, aspect=aspect
    )


@celery_app.task(
    name="app.workers.tasks.quality.run_quality_analysis",
    time_limit=QUALITY_TIME_LIMIT_S,
    soft_time_limit=QUALITY_SOFT_TIME_LIMIT_S,
)
def run_quality_analysis(dataset_id: str) -> dict:
    # Audit finding BE-10: unlike inference/training/roboflow, quality
    # analysis has no dedicated job row to mark FAILED — it's genuinely
    # fire-and-forget from the API's point of view (analyze-quality just
    # returns a task_id nothing currently polls). A real fix is tracking
    # this the same way the other job types are (its own table), which is
    # a bigger schema change than this pass takes on; in the meantime, a
    # crash at least surfaces here instead of vanishing with zero trace —
    # check worker logs for "quality analysis failed" if a dataset's
    # analysis seems to have silently done nothing.
    db = SessionLocal()
    try:
        images = list(db.scalars(select(Image).where(Image.dataset_id == uuid.UUID(dataset_id))))
        if not images:
            return {"analyzed": 0}

        project = db.get(Project, images[0].project_id)

        # Cone-like classes get "static" tracker treatment; nothing is
        # assumed "fast" (ball-like) here since this pass is about
        # anomaly detection, not smoothing — a wider static set costs
        # nothing but a fast-object misclassified as static would.
        cone_like_ids = frozenset(
            entry["id"] for entry in (project.class_config or []) if "cone" in entry.get("name", "").lower()
        )

        by_video: dict[uuid.UUID, list[Image]] = {}
        standalone: list[Image] = []
        for image in images:
            if image.video_id is not None:
                by_video.setdefault(image.video_id, []).append(image)
            else:
                standalone.append(image)

        analyzed = 0
        for video_id in by_video:
            track_stats = run_temporal_pass_for_video(db, video_id=video_id, static_class_ids=cone_like_ids)
            for image in by_video[video_id]:
                _ensure_pose_context(db, image, project)
                # annotation ids are globally unique, so the whole video's
                # track-stats map can be passed as-is — analyze_image_quality
                # only looks up the ids that belong to this image.
                analyze_image_quality(
                    db, image=image, project=project, is_video_frame=True, track_stats_by_annotation_id=track_stats
                )
                analyzed += 1

        for image in standalone:
            _ensure_pose_context(db, image, project)
            analyze_image_quality(db, image=image, project=project, is_video_frame=False)
            analyzed += 1

        return {"analyzed": analyzed}
    except Exception:
        logger.exception("quality analysis failed for dataset_id=%s", dataset_id)
        raise
    finally:
        db.close()
