from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.ml_model import ModelKind
from app.schemas.project import ClassEntry


class ModelRegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    weights_path: str = Field(description="Absolute path to the weights file, under ARTIFACTS_DIR")
    kind: ModelKind
    version: str = "v1"
    framework: str = "ultralytics"


class ModelRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    version: str
    kind: ModelKind
    framework: str
    weights_path: str
    class_config: list[ClassEntry]
    metrics: dict
    base_model_id: uuid.UUID | None
    is_active: bool
    created_at: datetime
    updated_at: datetime
