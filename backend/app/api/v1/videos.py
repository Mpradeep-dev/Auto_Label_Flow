from __future__ import annotations

import uuid

import cv2
from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import UploadKind, safe_storage_key, stream_upload_to_temp, validate_extension
from app.db.session import get_db
from app.models.dataset import Dataset
from app.models.video import Video
from app.schemas.video import FrameSampleConfig, VideoRead
from app.services.storage.factory import get_storage
from app.workers.tasks.video import extract_video_frames

router = APIRouter(tags=["videos"])


def _get_dataset_or_404(dataset_id: uuid.UUID, db: Session) -> Dataset:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found")
    return dataset


def _to_read(video: Video) -> VideoRead:
    return VideoRead(
        id=video.id,
        project_id=video.project_id,
        dataset_id=video.dataset_id,
        original_filename=video.original_filename,
        width=video.width,
        height=video.height,
        fps=video.fps,
        duration_s=video.duration_s,
        total_frames=video.total_frames,
        status=video.status,
        extracted_frame_count=video.extracted_frame_count,
        error=video.error,
        created_at=video.created_at,
        url=get_storage().get_url(video.storage_key),
    )


@router.post("/datasets/{dataset_id}/videos", response_model=VideoRead, status_code=status.HTTP_201_CREATED)
async def upload_video(dataset_id: uuid.UUID, file: UploadFile, db: Session = Depends(get_db)) -> VideoRead:
    dataset = _get_dataset_or_404(dataset_id, db)
    ext = validate_extension(file.filename or "upload", UploadKind.VIDEO)

    tmp_path = await stream_upload_to_temp(file, ext)
    try:
        cap = cv2.VideoCapture(str(tmp_path))
        if not cap.isOpened():
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "File is not a readable video")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        cap.release()
        duration_s = (total_frames / fps) if (total_frames and fps) else None

        key = safe_storage_key(
            str(dataset.project_id), str(dataset_id), "videos", original_filename=file.filename or "upload"
        )
        get_storage().upload(tmp_path, key, content_type=file.content_type)
    finally:
        tmp_path.unlink(missing_ok=True)

    video = Video(
        project_id=dataset.project_id,
        dataset_id=dataset_id,
        storage_key=key,
        original_filename=file.filename or "upload",
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames,
        duration_s=duration_s,
    )
    db.add(video)
    db.commit()
    db.refresh(video)
    return _to_read(video)


@router.get("/datasets/{dataset_id}/videos", response_model=list[VideoRead])
def list_videos(dataset_id: uuid.UUID, db: Session = Depends(get_db)) -> list[VideoRead]:
    _get_dataset_or_404(dataset_id, db)
    rows = list(db.scalars(select(Video).where(Video.dataset_id == dataset_id).order_by(Video.created_at.desc())))
    return [_to_read(v) for v in rows]


@router.get("/videos/{video_id}", response_model=VideoRead)
def get_video(video_id: uuid.UUID, db: Session = Depends(get_db)) -> VideoRead:
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Video not found")
    return _to_read(video)


@router.post("/videos/{video_id}/extract-frames", response_model=VideoRead, status_code=status.HTTP_202_ACCEPTED)
def trigger_extract_frames(
    video_id: uuid.UUID, config: FrameSampleConfig = FrameSampleConfig(), db: Session = Depends(get_db)
) -> VideoRead:
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Video not found")

    extract_video_frames.delay(str(video_id), config.interval, config.fps)
    db.refresh(video)  # in test/eager mode the task has already run synchronously by this point
    return _to_read(video)


@router.delete("/videos/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_video(video_id: uuid.UUID, db: Session = Depends(get_db)) -> None:
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Video not found")
    get_storage().delete(video.storage_key)
    db.delete(video)
    db.commit()
