from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.annotation import AnnotationReviewStatus, AnnotationSource, ErrorCategory, ErrorReason, ShapeType

_Point = tuple[float, float]


class AnnotationCreate(BaseModel):
    image_id: uuid.UUID
    class_id: int
    class_name: str
    shape_type: ShapeType = ShapeType.BBOX
    points: list[_Point] | None = None
    x1: float | None = Field(default=None, ge=0.0, le=1.0)
    y1: float | None = Field(default=None, ge=0.0, le=1.0)
    x2: float | None = Field(default=None, ge=0.0, le=1.0)
    y2: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_geometry(self) -> "AnnotationCreate":
        if self.shape_type == ShapeType.POLYGON:
            if not self.points or len(self.points) < 3:
                raise ValueError("A polygon needs at least 3 points")
            for x, y in self.points:
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    raise ValueError("Polygon points must be normalized in [0, 1]")
        else:
            if self.x1 is None or self.y1 is None or self.x2 is None or self.y2 is None:
                raise ValueError("x1, y1, x2, y2 are required for a BBOX annotation")
            if self.x2 <= self.x1 or self.y2 <= self.y1:
                raise ValueError("x2 must be > x1 and y2 must be > y1")
        return self


class AnnotationUpdate(BaseModel):
    class_id: int | None = None
    class_name: str | None = None
    points: list[_Point] | None = None
    x1: float | None = Field(default=None, ge=0.0, le=1.0)
    y1: float | None = Field(default=None, ge=0.0, le=1.0)
    x2: float | None = Field(default=None, ge=0.0, le=1.0)
    y2: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_geometry(self) -> "AnnotationUpdate":
        if self.points is not None and len(self.points) < 3:
            raise ValueError("A polygon needs at least 3 points")
        if (self.x1 is not None and self.x2 is not None and self.x2 <= self.x1) or (
            self.y1 is not None and self.y2 is not None and self.y2 <= self.y1
        ):
            raise ValueError("x2 must be > x1 and y2 must be > y1")
        return self


class AnnotationDelete(BaseModel):
    error_category: ErrorCategory | None = None
    error_reason: ErrorReason | None = None


class AnnotationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    image_id: uuid.UUID
    class_id: int
    class_name: str
    shape_type: ShapeType
    points: list[_Point] | None
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
