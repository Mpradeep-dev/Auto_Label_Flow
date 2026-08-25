from __future__ import annotations

import uuid
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.image import Image, ImageReviewStatus
from app.models.project import Project
from app.models.quality import AnnotationFlag, FlagType
from app.schemas.quality import (
    AnnotationFlagRead,
    FlagResolveRequest,
    ReviewQueueItem,
    ReviewQueuePage,
)
from app.services.inference.pose_context import compute_and_store_pose_context
from app.services.quality.run_analysis import analyze_image_quality
from app.services.storage.factory import get_storage
from app.workers.tasks.quality import run_quality_analysis

router = APIRouter(tags=["quality"])


def _get_image_or_404(image_id: uuid.UUID, db: Session) -> Image:
    image = db.get(Image, image_id)
    if image is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image not found")
    return image


@router.post("/images/{image_id}/analyze-quality", response_model=list[AnnotationFlagRead])
def analyze_image_quality_endpoint(image_id: uuid.UUID, db: Session = Depends(get_db)) -> list[AnnotationFlag]:
    image = _get_image_or_404(image_id, db)
    project = db.get(Project, image.project_id)

    if project.pose_model_id is not None:
        data = get_storage().read_bytes(image.storage_key)
        arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is not None:
            aspect = image.width / image.height if image.height else 1.0
            compute_and_store_pose_context(
                db, image_id=image.id, pose_model_id=project.pose_model_id, image_bgr=arr, aspect=aspect
            )

    is_video_frame = image.video_id is not None
    return analyze_image_quality(db, image=image, project=project, is_video_frame=is_video_frame)


@router.get("/images/{image_id}/flags", response_model=list[AnnotationFlagRead])
def list_image_flags(image_id: uuid.UUID, db: Session = Depends(get_db)) -> list[AnnotationFlag]:
    _get_image_or_404(image_id, db)
    return list(db.scalars(select(AnnotationFlag).where(AnnotationFlag.image_id == image_id)))


@router.post("/annotation-flags/{flag_id}/resolve", response_model=AnnotationFlagRead)
def resolve_flag(flag_id: uuid.UUID, payload: FlagResolveRequest, db: Session = Depends(get_db)) -> AnnotationFlag:
    flag = db.get(AnnotationFlag, flag_id)
    if flag is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Flag not found")
    flag.resolution = payload.resolution
    flag.resolved_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(flag)
    return flag


@router.post("/datasets/{dataset_id}/analyze-quality", status_code=status.HTTP_202_ACCEPTED)
def analyze_dataset_quality(dataset_id: uuid.UUID) -> dict:
    result = run_quality_analysis.delay(str(dataset_id))
    return {"task_id": getattr(result, "id", None)}


@router.get("/review/queue", response_model=ReviewQueuePage)
def get_review_queue(
    project_id: uuid.UUID = Query(...),
    dataset_id: uuid.UUID | None = None,
    flag_type: FlagType | None = None,
    review_status: ImageReviewStatus | None = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
) -> ReviewQueuePage:
    """Prioritized by difficulty_score DESC (nulls last — an image with no
    computed score yet isn't assumed easy, it's just unranked, so it sorts
    after ranked images but images are never hidden for lacking a score),
    except for the APPROVED bucket, which has nothing left to prioritize by
    difficulty — most-recently-approved-first there instead, so the queue
    reads as "here's what a human just cleared," not a frozen snapshot."""
    limit = max(1, min(limit, 200))

    query = select(Image).where(Image.project_id == project_id)
    if dataset_id is not None:
        query = query.where(Image.dataset_id == dataset_id)
    if flag_type is not None:
        flagged_image_ids = select(AnnotationFlag.image_id).where(AnnotationFlag.flag_type == flag_type)
        query = query.where(Image.id.in_(flagged_image_ids))
    if review_status is not None:
        query = query.where(Image.review_status == review_status)

    total = db.scalar(select(func.count()).select_from(query.subquery())) or 0

    if review_status == ImageReviewStatus.APPROVED:
        ordered = query.order_by(Image.updated_at.desc())
    else:
        ordered = query.order_by(Image.difficulty_score.desc().nullslast(), Image.created_at.asc())
    ordered = ordered.limit(limit).offset(offset)
    images = list(db.scalars(ordered))

    items = []
    for image in images:
        flags = list(db.scalars(select(AnnotationFlag).where(AnnotationFlag.image_id == image.id)))
        items.append(
            ReviewQueueItem(
                image_id=image.id,
                dataset_id=image.dataset_id,
                url=get_storage().get_url(image.storage_key),
                difficulty_score=image.difficulty_score,
                review_status=image.review_status.value,
                flags=flags,
            )
        )

    return ReviewQueuePage(items=items, total=total, limit=limit, offset=offset)
