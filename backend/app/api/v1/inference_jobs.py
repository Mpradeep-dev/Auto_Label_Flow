from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.dataset import Dataset
from app.models.inference_job import InferenceJob, JobStatus
from app.schemas.inference_job import InferenceJobCreate, InferenceJobRead
from app.workers.progress import get_progress, request_cancel
from app.workers.tasks.inference import run_inference_batch

router = APIRouter(prefix="/inference/jobs", tags=["inference"])

_TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED"}


@router.post("", response_model=InferenceJobRead, status_code=status.HTTP_202_ACCEPTED)
def create_inference_job(payload: InferenceJobCreate, db: Session = Depends(get_db)) -> InferenceJob:
    dataset = db.get(Dataset, payload.dataset_id)
    if dataset is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found")

    # Cheap pre-check for a clean, specific error message on the common
    # case; the partial unique index below (`uq_inference_jobs_one_active_
    # per_dataset`, see InferenceJob.__table_args__) is what actually
    # closes the race for two truly concurrent requests — this check alone
    # would still have a TOCTOU gap between two simultaneous callers.
    existing = db.scalar(
        select(InferenceJob).where(
            InferenceJob.dataset_id == payload.dataset_id,
            InferenceJob.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]),
        )
    )
    if existing is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"An inference job ({existing.id}) is already {existing.status.value.lower()} for this dataset — "
            "wait for it to finish, or cancel it, before starting another.",
        )

    job = InferenceJob(
        project_id=dataset.project_id,
        dataset_id=payload.dataset_id,
        model_id=payload.model_id,
        status=JobStatus.QUEUED,
        conf=payload.conf,
        iou=payload.iou,
    )
    db.add(job)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Another inference job was just started for this dataset — wait for it to finish before starting another.",
        ) from exc
    db.refresh(job)

    run_inference_batch.delay(str(job.id))
    db.refresh(job)  # eager/test mode: already terminal by the time we return
    return job


@router.get("", response_model=list[InferenceJobRead])
def list_inference_jobs(project_id: uuid.UUID, db: Session = Depends(get_db)) -> list[InferenceJob]:
    """Full run history for a project — unlike /latest (scoped to one
    dataset, and only useful for reattaching to a still-live job), this
    backs a persistent history view so a job that finished (or failed)
    while the user was on another page doesn't just disappear."""
    return list(
        db.scalars(
            select(InferenceJob).where(InferenceJob.project_id == project_id).order_by(InferenceJob.created_at.desc())
        )
    )


@router.get("/latest", response_model=InferenceJobRead | None)
def get_latest_inference_job(dataset_id: uuid.UUID, db: Session = Depends(get_db)) -> InferenceJob | None:
    """Lets the Auto Annotation page reattach to a job it kicked off
    before a navigation away (or a reload) wiped its local state —
    otherwise a still-running batch just disappears from the UI even
    though it keeps going server-side. Must be registered before
    `/{job_id}` — Starlette matches routes in registration order and
    `{job_id}: uuid.UUID` would otherwise swallow "latest" first and 422
    trying to parse it as a UUID (see the identical Roboflow-jobs gotcha)."""
    return db.scalar(
        select(InferenceJob).where(InferenceJob.dataset_id == dataset_id).order_by(InferenceJob.created_at.desc()).limit(1)
    )


@router.get("/{job_id}", response_model=InferenceJobRead)
def get_inference_job(job_id: uuid.UUID, db: Session = Depends(get_db)) -> InferenceJob:
    job = db.get(InferenceJob, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    return job


@router.post("/{job_id}/cancel", response_model=InferenceJobRead)
def cancel_inference_job(job_id: uuid.UUID, db: Session = Depends(get_db)) -> InferenceJob:
    job = db.get(InferenceJob, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    request_cancel(str(job_id))
    return job


@router.get("/{job_id}/stream")
async def stream_inference_job(job_id: uuid.UUID, db: Session = Depends(get_db)):
    """SSE progress stream (PLAN "SSE over WebSockets because the flow is
    one-directional and reconnects itself"). Polls the Redis progress key
    written by ThrottledProgressWriter; falls back to the DB row's terminal
    state if Redis has nothing (e.g. a job that finished before its TTL, or
    ran so fast progress was never throttled-written)."""
    job = db.get(InferenceJob, job_id)
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
                fresh = db.get(InferenceJob, job_id)
                payload = json.dumps(
                    {
                        "current": fresh.processed_images if fresh else 0,
                        "total": fresh.total_images if fresh else 0,
                        "predictions": fresh.total_predictions if fresh else 0,
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

            if status_value in _TERMINAL_STATUSES:
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
