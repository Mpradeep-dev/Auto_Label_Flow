from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.quality import FlagResolution, FlagType


class AnnotationFlagRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    annotation_id: uuid.UUID
    image_id: uuid.UUID
    flag_type: FlagType
    severity: float
    reason: str
    details: dict
    resolution: FlagResolution | None
    created_at: datetime


class FlagResolveRequest(BaseModel):
    resolution: FlagResolution


class ReviewQueueItem(BaseModel):
    image_id: uuid.UUID
    dataset_id: uuid.UUID
    url: str
    difficulty_score: float | None
    review_status: str
    flags: list[AnnotationFlagRead]


class ReviewQueuePage(BaseModel):
    items: list[ReviewQueueItem]
    total: int
    limit: int
    offset: int
