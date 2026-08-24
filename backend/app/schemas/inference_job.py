from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.inference_job import JobStatus


class InferenceJobCreate(BaseModel):
    dataset_id: uuid.UUID
    model_id: uuid.UUID
    conf: float = Field(default=0.20, ge=0.0, le=1.0)
    iou: float = Field(default=0.70, ge=0.0, le=1.0)


class InferenceJobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID
    dataset_id: uuid.UUID
    model_id: uuid.UUID
    status: JobStatus
    conf: float
    iou: float
    total_images: int
    processed_images: int
    failed_images: int
    total_predictions: int
    error: str | None
    created_at: datetime
    updated_at: datetime


class JobProgressRead(BaseModel):
    current: int
    total: int
    predictions: int
    fps: float
    eta_s: float | None
    status: str
    error: str | None = None
