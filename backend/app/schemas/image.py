from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.image import ImageReviewStatus, ImageSourceType


class ImageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID
    dataset_id: uuid.UUID
    original_filename: str
    width: int
    height: int
    source_type: ImageSourceType
    video_id: uuid.UUID | None
    frame_index: int | None
    frame_timestamp_s: float | None
    review_status: ImageReviewStatus
    difficulty_score: float | None
    created_at: datetime
    url: str  # populated in the route from ObjectStorage.get_url, not a DB column


class ImageListPage(BaseModel):
    items: list[ImageRead]
    total: int
    limit: int
    offset: int
