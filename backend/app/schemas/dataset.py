from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.dataset import DatasetStatus


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = None


class DatasetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID
    name: str
    description: str | None
    status: DatasetStatus
    created_at: datetime
    updated_at: datetime


class DatasetStats(BaseModel):
    """Cheap counts for the dataset list/detail view — deliberately not the
    full statistics dashboard (that's Phase 8), just enough to orient."""

    total_images: int
    pending_images: int
    approved_images: int
    total_videos: int
