"""Request/response shapes for the Settings-page integrations panel
(PLAN follow-on: Kaggle + Roboflow connect). Never echoes a stored secret
back in full — `identifier` is the only thing a GET reveals."""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.roboflow_job import RoboflowJobKind, RoboflowJobStatus


class IntegrationStatus(BaseModel):
    provider: str
    connected: bool
    identifier: str | None = None
    verified_at: datetime | None = None
    last_error: str | None = None


class KaggleConnectRequest(BaseModel):
    username: str = Field(min_length=1)
    key: str = Field(min_length=1)


class ModalConnectRequest(BaseModel):
    token_id: str = Field(min_length=1)
    token_secret: str = Field(min_length=1)


class RoboflowConnectRequest(BaseModel):
    api_key: str = Field(min_length=1)
    default_workspace: str | None = None


class RoboflowExportRequest(BaseModel):
    workspace: str = Field(min_length=1)
    project: str = Field(min_length=1)


class RoboflowExportResult(BaseModel):
    uploaded: int
    failed: int
    failures: list[str]


class RoboflowImportRequest(BaseModel):
    workspace: str = Field(min_length=1)
    project: str = Field(min_length=1)
    # None means "no generated Version" — pull the project's raw uploaded
    # images instead (see `import_roboflow_raw_project`).
    version: int | None = None
    dataset_name: str | None = None
    # Raw pull only (ignored when `version` is set): narrow to images with
    # zero existing Roboflow annotations.
    unannotated_only: bool = False


class RoboflowProjectSummary(BaseModel):
    workspace: str
    project: str
    name: str
    type: str
    image_count: int


class RoboflowVersionSummary(BaseModel):
    version: int
    image_count: int


class RoboflowJobRead(BaseModel):
    """Background job row backing the import/export progress bars — polled
    directly and streamed over SSE (see `integrations.py`'s `/jobs/{id}` and
    `/jobs/{id}/stream`, mirroring `inference_jobs.py`)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID
    kind: RoboflowJobKind
    status: RoboflowJobStatus
    workspace: str
    project_slug: str
    version: int | None = None
    unannotated_only: bool
    total_items: int
    processed_items: int
    uploaded_count: int
    failed_count: int
    failures: list[str]
    result_dataset_id: uuid.UUID | None = None
    dataset_version_id: uuid.UUID | None = None
    error: str | None = None
    created_at: datetime
