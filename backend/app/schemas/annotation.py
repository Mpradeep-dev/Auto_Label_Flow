from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.annotation import AnnotationReviewStatus, AnnotationSource, ErrorCategory, ErrorReason


class AnnotationCreate(BaseModel):
    image_id: uuid.UUID
    class_id: int
    class_name: str
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class AnnotationUpdate(BaseModel):
    class_id: int | None = None
    class_name: str | None = None
    x1: float | None = Field(default=None, ge=0.0, le=1.0)
    y1: float | None = Field(default=None, ge=0.0, le=1.0)
    x2: float | None = Field(default=None, ge=0.0, le=1.0)
    y2: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class AnnotationDelete(BaseModel):
    error_category: ErrorCategory | None = None
    error_reason: ErrorReason | None = None


class AnnotationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    image_id: uuid.UUID
    class_id: int
    class_name: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float | None
    source: AnnotationSource
    review_status: AnnotationReviewStatus
    revision_seq: int
    created_at: datetime
    updated_at: datetime


class AutoAnnotateRequest(BaseModel):
    model_id: uuid.UUID
    conf: float = Field(default=0.20, ge=0.0, le=1.0)
    iou: float = Field(default=0.70, ge=0.0, le=1.0)
    replace_existing: bool = Field(
        default=True, description="Delete this image's current AUTO annotations before writing new ones"
    )
