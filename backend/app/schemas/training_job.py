from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.training_job import TrainingJobStatus, TrainingProviderType


class TrainingJobCreate(BaseModel):
    dataset_version_id: uuid.UUID
    base_model_id: uuid.UUID
    provider: TrainingProviderType = TrainingProviderType.LOCAL
    epochs: int = Field(default=100, ge=1, le=2000)
    batch_size: int = Field(default=8, ge=1, le=256)
    image_size: int = Field(default=640, ge=32, le=2048)
    learning_rate: float | None = Field(default=None, gt=0.0)
    device: str = "0"


class TrainingJobEpochRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    epoch: int
    box_loss: float | None
    cls_loss: float | None
    dfl_loss: float | None
    precision: float | None
    recall: float | None
    map50: float | None
    map50_95: float | None
    recorded_at: datetime


class TrainingJobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID
    dataset_version_id: uuid.UUID
    base_model_id: uuid.UUID | None
    result_model_id: uuid.UUID | None
    provider: TrainingProviderType
    status: TrainingJobStatus
    epochs: int
    batch_size: int
    image_size: int
    learning_rate: float | None
    device: str
    current_epoch: int
    metrics: dict
    error: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    failed_at: datetime | None


class GPUInfoRead(BaseModel):
    torch_version: str
    cuda_available: bool
    device_name: str | None
    vram_total_mb: float | None
    cuda_version: str | None


class TrainingProvidersRead(BaseModel):
    available: list[str]
    gpu: GPUInfoRead
