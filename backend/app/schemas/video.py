from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.video import VideoStatus


class FrameSampleConfig(BaseModel):
    """Video sampling config, POSTed to trigger extraction. Only one of
    `interval` (every N frames) or `fps` (N frames per second) should be
    set; `interval` wins if both are given. Default interval is denser
    than a naive "every 10 frames" because this corpus's clips are short
    (1.8-7s) — see PLAN 'Sample footage characteristics'."""

    interval: int | None = Field(default=None, ge=1, description="Extract every Nth frame")
    fps: float | None = Field(default=None, gt=0, description="Extract at this many frames per second")


class VideoRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID
    dataset_id: uuid.UUID
    original_filename: str
    width: int | None
    height: int | None
    fps: float | None
    duration_s: float | None
    total_frames: int | None
    status: VideoStatus
    extracted_frame_count: int
    error: str | None
    created_at: datetime
    url: str
