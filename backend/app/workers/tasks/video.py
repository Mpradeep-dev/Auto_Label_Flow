"""Frame extraction task — runs on the `default` queue (CPU/IO-bound, no
GPU involved), so it never contends with the `gpu` queue's concurrency=1
inference/training work."""
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import cv2

from app.core.security import safe_storage_key
from app.db.session import SessionLocal
from app.models.image import Image, ImageSourceType
from app.models.video import Video, VideoStatus
from app.services.storage.factory import get_storage
from app.services.video.sampler import compute_sample_indices
from app.workers.celery_app import VIDEO_SOFT_TIME_LIMIT_S, VIDEO_TIME_LIMIT_S, celery_app


@celery_app.task(
    bind=True,
    name="app.workers.tasks.video.extract_video_frames",
    time_limit=VIDEO_TIME_LIMIT_S,
    soft_time_limit=VIDEO_SOFT_TIME_LIMIT_S,
)
def extract_video_frames(self, video_id: str, interval: int | None = None, fps: float | None = None) -> None:
    db = SessionLocal()
    video = db.get(Video, uuid.UUID(video_id))
    if video is None:
        db.close()
        return

    video.status = VideoStatus.EXTRACTING
    db.commit()

    tmp_path: Path | None = None
    try:
        import os

        suffix = Path(video.original_filename).suffix or ".mp4"
        fd, tmp_name = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        tmp_path = Path(tmp_name)
        get_storage().download(video.storage_key, tmp_path)  # overwrites the empty temp file

        cap = cv2.VideoCapture(str(tmp_path))
        if not cap.isOpened():
            # Audit finding BE-18: without this check, a corrupt/unsupported
            # file falls straight through to `compute_sample_indices(0, ...)`
            # returning an empty index list, and the task marks the video
            # EXTRACTED with 0 frames — a silent no-op reported as success,
            # not the failure it actually is.
            cap.release()
            raise ValueError(
                f"Could not open {video.original_filename!r} as a video — the file may be corrupt "
                "or in an unsupported format."
            )
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        indices = compute_sample_indices(total_frames, video_fps, interval=interval, fps=fps)

        extracted = 0
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            success, buf = cv2.imencode(".jpg", frame)
            if not success:
                continue

            key = safe_storage_key(
                str(video.project_id),
                str(video.dataset_id),
                "frames",
                str(video.id),
                original_filename=f"frame_{idx:06d}.jpg",
            )
            get_storage().upload_bytes(buf.tobytes(), key, content_type="image/jpeg")

            frame_h, frame_w = frame.shape[:2]
            image = Image(
                project_id=video.project_id,
                dataset_id=video.dataset_id,
                storage_key=key,
                original_filename=f"frame_{idx:06d}.jpg",
                width=frame_w,
                height=frame_h,
                source_type=ImageSourceType.VIDEO_FRAME,
                video_id=video.id,
                frame_index=idx,
                frame_timestamp_s=(idx / video_fps) if video_fps > 0 else None,
            )
            db.add(image)
            extracted += 1

        cap.release()

        if indices and extracted == 0:
            # The file opened, but every single sampled frame failed to
            # decode — also a real failure, not a video that legitimately
            # has zero usable frames (BE-18).
            raise ValueError(
                f"{video.original_filename!r} opened but no frames could be read from it — "
                "the file is likely corrupt."
            )

        video.status = VideoStatus.EXTRACTED
        video.extracted_frame_count = extracted
        video.fps = video_fps
        video.total_frames = total_frames
        video.width = width or video.width
        video.height = height or video.height
        video.frame_extraction_config = {"interval": interval, "fps": fps}
        db.commit()
    except Exception as exc:
        db.rollback()
        video = db.get(Video, uuid.UUID(video_id))
        if video is not None:
            video.status = VideoStatus.FAILED
            video.error = str(exc)
            db.commit()
        raise
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        db.close()
