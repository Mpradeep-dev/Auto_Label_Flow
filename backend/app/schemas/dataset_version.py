from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.dataset_version import DatasetVersionStatus


class DatasetVersionCreate(BaseModel):
    train_ratio: float = Field(default=0.8, ge=0.0, le=1.0)
    val_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    test_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    seed: int = 0


class DatasetVersionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    dataset_id: uuid.UUID
    version_number: int
    status: DatasetVersionStatus
    train_ratio: float
    val_ratio: float
    test_ratio: float
    split_seed: int
    used_frame_level_fallback: bool
    total_images: int
    total_annotations: int
    error: str | None
    download_url: str | None = None
    coco_download_url: str | None = None
    cvat_download_url: str | None = None
    created_at: datetime
