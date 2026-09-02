"""Request/response shapes for the Settings-page integrations panel
(PLAN follow-on: Kaggle + Roboflow connect). Never echoes a stored secret
back in full — `identifier` is the only thing a GET reveals."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.blob_import_job import BlobImportJobStatus
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
    # Raw pull only (ignored when `version` is set): narrow to one upload
    # batch (from `RoboflowBatchSummary.id`) instead of every raw image.
    batch_id: str | None = None


class RoboflowProjectSummary(BaseModel):
    workspace: str
    project: str
    name: str
    type: str
    image_count: int


class RoboflowVersionSummary(BaseModel):
    version: int
    image_count: int


class RoboflowBatchSummary(BaseModel):
    id: str
    name: str
    image_count: int


class BlobImportRequest(BaseModel):
    """Kick off an "Import from Azure Blob" job: a prefix inside the app's
    configured container, plus how its sibling labels are laid out."""

    prefix: str = Field(min_length=1)
    label_format: Literal["auto", "yolo", "coco"] = "auto"
    dataset_name: str | None = None


class BlobImportJobRead(BaseModel):
    """Background job row backing the Azure-Blob import progress bar —
    polled and SSE-streamed exactly like `RoboflowJobRead`."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID
    status: BlobImportJobStatus
    prefix: str
    label_format: str
    dataset_name: str | None = None
    total_items: int
    processed_items: int
    result_dataset_id: uuid.UUID | None = None
    error: str | None = None
    created_at: datetime


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
    batch_id: str | None = None
    total_items: int
    processed_items: int
    uploaded_count: int
    failed_count: int
    failures: list[str]
    result_dataset_id: uuid.UUID | None = None
    dataset_version_id: uuid.UUID | None = None
    error: str | None = None
    created_at: datetime
