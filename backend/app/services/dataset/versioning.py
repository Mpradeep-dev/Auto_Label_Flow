"""Creates a DatasetVersion: selects reviewed (APPROVED) images, splits
them by source video, and pins each image's current annotations to their
latest event — see `app/models/dataset_version.py` for why pins over
copy-on-create snapshots."""
from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.annotation import Annotation
from app.models.dataset_version import (
    DatasetVersion,
    DatasetVersionAnnotationPin,
    DatasetVersionImage,
    DatasetVersionStatus,
    SplitName,
)
from app.models.image import Image, ImageReviewStatus
from app.services.dataset.splitter import ImageGroupInfo, split_images


class NoApprovedImagesError(ValueError):
    pass


def create_version(
    db: Session,
    *,
    dataset_id: uuid.UUID,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 0,
) -> DatasetVersion:
    images = list(
        db.scalars(
            select(Image).where(Image.dataset_id == dataset_id, Image.review_status == ImageReviewStatus.APPROVED)
        )
    )
    if not images:
        raise NoApprovedImagesError(
            "No approved images in this dataset — approve at least one image before creating a version."
        )

    group_infos = [
        ImageGroupInfo(
            image_id=str(img.id),
            source_group_id=str(img.video_id) if img.video_id else f"image:{img.id}",
        )
        for img in images
    ]
    result = split_images(group_infos, train_ratio, val_ratio, test_ratio, seed=seed)

    next_version_number = (
        db.scalar(select(func.max(DatasetVersion.version_number)).where(DatasetVersion.dataset_id == dataset_id))
        or 0
    ) + 1

    version = DatasetVersion(
        dataset_id=dataset_id,
        version_number=next_version_number,
        status=DatasetVersionStatus.DRAFT,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split_seed=seed,
        used_frame_level_fallback=result.used_frame_level_fallback,
        total_images=len(images),
    )
    db.add(version)
    db.flush()  # need version.id for the child rows below

    group_by_id = {str(img.id): img for img in images}
    total_annotations = 0
    for image_id_str, split_value in result.assignment.items():
        img = group_by_id[image_id_str]
        db.add(
            DatasetVersionImage(
                dataset_version_id=version.id,
                image_id=img.id,
                split=SplitName(split_value),
                source_group_id=str(img.video_id) if img.video_id else f"image:{img.id}",
            )
        )
        annotations = list(db.scalars(select(Annotation).where(Annotation.image_id == img.id)))
        for ann in annotations:
            db.add(
                DatasetVersionAnnotationPin(
                    dataset_version_id=version.id,
                    annotation_id=ann.id,
                    pinned_event_id=ann.latest_event_id,
                    image_id=img.id,
                )
            )
        total_annotations += len(annotations)

    version.total_annotations = total_annotations
    db.commit()
    db.refresh(version)
    return version
