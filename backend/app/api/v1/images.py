from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.security import UploadKind, safe_storage_key, stream_upload_to_temp, validate_extension
from app.db.session import get_db
from app.models.dataset import Dataset
from app.models.image import Image, ImageReviewStatus
from app.schemas.image import ImageListPage, ImageRead
from app.services.storage.factory import get_storage

router = APIRouter(tags=["images"])


def _get_dataset_or_404(dataset_id: uuid.UUID, db: Session) -> Dataset:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found")
    return dataset


def _to_read(image: Image) -> ImageRead:
    return ImageRead(
        id=image.id,
        project_id=image.project_id,
        dataset_id=image.dataset_id,
        original_filename=image.original_filename,
        width=image.width,
        height=image.height,
        source_type=image.source_type,
        video_id=image.video_id,
        frame_index=image.frame_index,
        frame_timestamp_s=image.frame_timestamp_s,
        review_status=image.review_status,
        difficulty_score=image.difficulty_score,
        created_at=image.created_at,
        url=get_storage().get_url(image.storage_key),
    )


@router.post(
    "/datasets/{dataset_id}/images",
    response_model=ImageRead,
    status_code=status.HTTP_201_CREATED,
)
async def upload_image(dataset_id: uuid.UUID, file: UploadFile, db: Session = Depends(get_db)) -> ImageRead:
    dataset = _get_dataset_or_404(dataset_id, db)
    ext = validate_extension(file.filename or "upload", UploadKind.IMAGE)

    tmp_path = await stream_upload_to_temp(file, ext)
    try:
        import cv2

        img = cv2.imread(str(tmp_path))
        if img is None:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "File is not a readable image")
        height, width = img.shape[:2]

        key = safe_storage_key(
            str(dataset.project_id), str(dataset_id), "images", original_filename=file.filename or "upload"
        )
        get_storage().upload(tmp_path, key, content_type=file.content_type)
    finally:
        tmp_path.unlink(missing_ok=True)

    image = Image(
        project_id=dataset.project_id,
        dataset_id=dataset_id,
        storage_key=key,
        original_filename=file.filename or "upload",
        width=width,
        height=height,
    )
    db.add(image)
    db.commit()
    db.refresh(image)
    return _to_read(image)


@router.get("/datasets/{dataset_id}/images", response_model=ImageListPage)
def list_images(
    dataset_id: uuid.UUID,
    limit: int = 50,
    offset: int = 0,
    review_status: ImageReviewStatus | None = None,
    db: Session = Depends(get_db),
) -> ImageListPage:
    _get_dataset_or_404(dataset_id, db)
    limit = max(1, min(limit, 200))

    conditions = [Image.dataset_id == dataset_id]
    if review_status is not None:
        conditions.append(Image.review_status == review_status)

    total = db.scalar(select(func.count()).select_from(Image).where(*conditions)) or 0
    rows = list(
        db.scalars(select(Image).where(*conditions).order_by(Image.created_at.asc()).limit(limit).offset(offset))
    )
    return ImageListPage(items=[_to_read(row) for row in rows], total=total, limit=limit, offset=offset)


@router.get("/images/{image_id}", response_model=ImageRead)
def get_image(image_id: uuid.UUID, db: Session = Depends(get_db)) -> ImageRead:
    image = db.get(Image, image_id)
    if image is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image not found")
    return _to_read(image)


@router.delete("/images/{image_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_image(image_id: uuid.UUID, db: Session = Depends(get_db)) -> None:
    image = db.get(Image, image_id)
    if image is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image not found")
    get_storage().delete(image.storage_key)
    db.delete(image)
    db.commit()
