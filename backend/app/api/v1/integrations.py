"""Settings-page endpoints: connect/disconnect Kaggle and Roboflow
(PLAN follow-on). Kept as one small router rather than folding into
`training_jobs.py` — these are account-level connections, not training
concepts, and Roboflow has nothing to do with training at all."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.blob_import_job import BlobImportJob
from app.models.roboflow_job import RoboflowJob, RoboflowJobKind
from app.schemas.integration import (
    BlobImportJobRead,
    IntegrationStatus,
    KaggleConnectRequest,
    ModalConnectRequest,
    RoboflowBatchSummary,
    RoboflowConnectRequest,
    RoboflowJobRead,
    RoboflowProjectSummary,
    RoboflowVersionSummary,
)
from app.services.integrations import kaggle_connect, modal_connect, roboflow_browse, roboflow_connect
from app.services.integrations.kaggle_connect import KaggleVerificationError
from app.services.integrations.modal_connect import ModalVerificationError
from app.services.integrations.roboflow_connect import RoboflowNotConnectedError, RoboflowVerificationError
from app.workers.progress import get_progress, request_cancel

router = APIRouter(prefix="/integrations", tags=["integrations"])
logger = logging.getLogger(__name__)


@router.get("", response_model=list[IntegrationStatus])
def list_integrations(db: Session = Depends(get_db)) -> list[IntegrationStatus]:
    return [kaggle_connect.get_status(db), modal_connect.get_status(db), roboflow_connect.get_status(db)]


@router.post("/kaggle", response_model=IntegrationStatus)
def connect_kaggle(payload: KaggleConnectRequest, db: Session = Depends(get_db)) -> IntegrationStatus:
    try:
        return kaggle_connect.connect(db, username=payload.username, key=payload.key)
    except KaggleVerificationError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED, "The `kaggle` package is not installed on this server"
        ) from exc


@router.delete("/kaggle", status_code=status.HTTP_204_NO_CONTENT)
def disconnect_kaggle(db: Session = Depends(get_db)) -> None:
    kaggle_connect.disconnect(db)


@router.post("/modal", response_model=IntegrationStatus)
def connect_modal(payload: ModalConnectRequest, db: Session = Depends(get_db)) -> IntegrationStatus:
    try:
        return modal_connect.connect(db, token_id=payload.token_id, token_secret=payload.token_secret)
    except ModalVerificationError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED, "The `modal` package is not installed on this server"
        ) from exc


@router.delete("/modal", status_code=status.HTTP_204_NO_CONTENT)
def disconnect_modal(db: Session = Depends(get_db)) -> None:
    modal_connect.disconnect(db)


@router.post("/roboflow", response_model=IntegrationStatus)
def connect_roboflow(payload: RoboflowConnectRequest, db: Session = Depends(get_db)) -> IntegrationStatus:
    try:
        return roboflow_connect.connect(db, api_key=payload.api_key, default_workspace=payload.default_workspace)
    except RoboflowVerificationError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED, "The `roboflow` package is not installed on this server"
        ) from exc


@router.delete("/roboflow", status_code=status.HTTP_204_NO_CONTENT)
def disconnect_roboflow(db: Session = Depends(get_db)) -> None:
    roboflow_connect.disconnect(db)


@router.get("/roboflow/projects", response_model=list[RoboflowProjectSummary])
def list_roboflow_projects(workspace: str | None = None, db: Session = Depends(get_db)) -> list[RoboflowProjectSummary]:
    try:
        return roboflow_browse.list_projects(db, workspace=workspace)
    except RoboflowNotConnectedError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except Exception as exc:
        logger.exception("Could not list Roboflow projects")
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Could not list Roboflow projects: {exc}") from exc


@router.get("/roboflow/projects/{workspace}/{project}/versions", response_model=list[RoboflowVersionSummary])
def list_roboflow_versions(
    workspace: str, project: str, db: Session = Depends(get_db)
) -> list[RoboflowVersionSummary]:
    try:
        return roboflow_browse.list_versions(db, workspace=workspace, project_slug=project)
    except RoboflowNotConnectedError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Could not list versions for {project!r}: {exc}") from exc


@router.get("/roboflow/projects/{workspace}/{project}/batches", response_model=list[RoboflowBatchSummary])
def list_roboflow_batches(
    workspace: str, project: str, db: Session = Depends(get_db)
) -> list[RoboflowBatchSummary]:
    try:
        return roboflow_browse.list_batches(db, workspace=workspace, project_slug=project)
    except RoboflowNotConnectedError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Could not list batches for {project!r}: {exc}") from exc


_TERMINAL_JOB_STATUSES = {"COMPLETED", "FAILED", "CANCELLED"}


@router.get("/roboflow/jobs/latest", response_model=RoboflowJobRead | None)
def get_latest_roboflow_job(
    kind: RoboflowJobKind,
    project_id: uuid.UUID | None = None,
    dataset_version_id: uuid.UUID | None = None,
    db: Session = Depends(get_db),
) -> RoboflowJob | None:
    """Lets a page reattach to whatever job it last kicked off for this
    project (import) or version (export) — leaving the page and coming
    back, or a hard reload, otherwise has nothing to poll and the progress
    bar just vanishes even though the job keeps running server-side. Must
    be registered before `/roboflow/jobs/{job_id}` — Starlette matches
    routes in registration order and `{job_id}: uuid.UUID` would otherwise
    swallow "latest" first and 422 trying to parse it as a UUID."""
    if kind == RoboflowJobKind.IMPORT and project_id is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "project_id is required for kind=IMPORT")
    if kind == RoboflowJobKind.EXPORT and dataset_version_id is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "dataset_version_id is required for kind=EXPORT")

    query = select(RoboflowJob).where(RoboflowJob.kind == kind)
    if project_id is not None:
        query = query.where(RoboflowJob.project_id == project_id)
    if dataset_version_id is not None:
        query = query.where(RoboflowJob.dataset_version_id == dataset_version_id)
    query = query.order_by(RoboflowJob.created_at.desc()).limit(1)
    return db.scalar(query)


@router.get("/roboflow/jobs/{job_id}", response_model=RoboflowJobRead)
def get_roboflow_job(job_id: uuid.UUID, db: Session = Depends(get_db)) -> RoboflowJob:
    job = db.get(RoboflowJob, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    return job


@router.post("/roboflow/jobs/{job_id}/cancel", response_model=RoboflowJobRead)
def cancel_roboflow_job(job_id: uuid.UUID, db: Session = Depends(get_db)) -> RoboflowJob:
    """Sets the Redis cancel flag the running task checks between items
    (same mechanism as `inference_jobs.py`'s cancel) — the task notices at
    its next per-item check and stops there, so this returns immediately
    without waiting for that to happen."""
    job = db.get(RoboflowJob, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    request_cancel(str(job_id))
    return job


@router.get("/roboflow/jobs/{job_id}/stream")
async def stream_roboflow_job(job_id: uuid.UUID, db: Session = Depends(get_db)):
    """SSE progress stream for an import or export job — same shape and
    reasoning as `inference_jobs.py`'s `/stream`: poll the Redis progress
    key on a throttle, fall back to the DB row's terminal state if Redis
    has nothing (job finished too fast to ever get throttle-written, or its
    key TTL'd out)."""
    job = db.get(RoboflowJob, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")

    async def event_stream():
        last_payload = None
        for _ in range(3600):  # ~30 min ceiling at the 0.5s poll below, then the client reconnects
            progress = get_progress(str(job_id))
            if progress is not None:
                payload = json.dumps(progress.__dict__)
                status_value = progress.status
            else:
                fresh = db.get(RoboflowJob, job_id)
                payload = json.dumps(
                    {
                        "current": fresh.processed_items if fresh else 0,
                        "total": fresh.total_items if fresh else 0,
                        "predictions": 0,
                        "fps": 0.0,
                        "eta_s": None,
                        "status": fresh.status.value if fresh else "UNKNOWN",
                        "error": fresh.error if fresh else None,
                        "failed": fresh.failed_count if fresh else 0,
                    }
                )
                status_value = fresh.status.value if fresh else "UNKNOWN"

            if payload != last_payload:
                yield f"data: {payload}\n\n"
                last_payload = payload

            if status_value in _TERMINAL_JOB_STATUSES:
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# --- Azure Blob import jobs (mirrors /roboflow/jobs/*; see datasets.py's
#     POST /projects/{id}/import/azure-blob for the create side) ---


@router.get("/azure-blob/jobs/latest", response_model=BlobImportJobRead | None)
def get_latest_blob_import_job(
    project_id: uuid.UUID, db: Session = Depends(get_db)
) -> BlobImportJob | None:
    """Lets the Datasets page reattach to the last Azure-Blob import it
    kicked off for this project after a navigation away or a reload.
    Registered before `/azure-blob/jobs/{job_id}` so "latest" isn't parsed
    as a UUID (same reason as the Roboflow route above)."""
    return db.scalar(
        select(BlobImportJob)
        .where(BlobImportJob.project_id == project_id)
        .order_by(BlobImportJob.created_at.desc())
        .limit(1)
    )


@router.get("/azure-blob/jobs/{job_id}", response_model=BlobImportJobRead)
def get_blob_import_job(job_id: uuid.UUID, db: Session = Depends(get_db)) -> BlobImportJob:
    job = db.get(BlobImportJob, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    return job


@router.post("/azure-blob/jobs/{job_id}/cancel", response_model=BlobImportJobRead)
def cancel_blob_import_job(job_id: uuid.UUID, db: Session = Depends(get_db)) -> BlobImportJob:
    """Sets the Redis cancel flag the running task checks before each image
    — same mechanism as the Roboflow cancel above."""
    job = db.get(BlobImportJob, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    request_cancel(str(job_id))
    return job


@router.get("/azure-blob/jobs/{job_id}/stream")
async def stream_blob_import_job(job_id: uuid.UUID, db: Session = Depends(get_db)):
    """SSE progress stream — same shape and fallback reasoning as
    `stream_roboflow_job`."""
    job = db.get(BlobImportJob, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")

    async def event_stream():
        last_payload = None
        for _ in range(3600):
            progress = get_progress(str(job_id))
            if progress is not None:
                payload = json.dumps(progress.__dict__)
                status_value = progress.status
            else:
                fresh = db.get(BlobImportJob, job_id)
                payload = json.dumps(
                    {
                        "current": fresh.processed_items if fresh else 0,
                        "total": fresh.total_items if fresh else 0,
                        "predictions": 0,
                        "fps": 0.0,
                        "eta_s": None,
                        "status": fresh.status.value if fresh else "UNKNOWN",
                        "error": fresh.error if fresh else None,
                    }
                )
                status_value = fresh.status.value if fresh else "UNKNOWN"

            if payload != last_payload:
                yield f"data: {payload}\n\n"
                last_payload = payload

            if status_value in _TERMINAL_JOB_STATUSES:
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
