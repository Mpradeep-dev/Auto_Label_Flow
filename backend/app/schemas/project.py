from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ClassEntry(BaseModel):
    """One class in a project's active taxonomy. `id` matches the
    underlying model's class index (`model.names` key) — colour is never
    stored here, the frontend derives it deterministically from list
    position (`classColors.ts`)."""

    id: int
    name: str


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = None


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = None
    class_config: list[ClassEntry] | None = None
    quality_rule_config: dict | None = None
    primary_model_id: uuid.UUID | None = None
    pose_model_id: uuid.UUID | None = None


class ProjectRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    slug: str
    description: str | None
    class_config: list[ClassEntry]
    quality_rule_config: dict
    primary_model_id: uuid.UUID | None
    pose_model_id: uuid.UUID | None
    created_at: datetime
    updated_at: datetime
