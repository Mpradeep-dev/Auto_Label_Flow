from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.dataset_version import DatasetVersion
from app.models.training_job import TrainingJob, TrainingJobEpoch, TrainingJobStatus
from app.schemas.training_job import (
    GPUInfoRead,
    TrainingJobCreate,
    TrainingJobEpochRead,
    TrainingJobRead,
    TrainingProvidersRead,
)
from app.services.training.gpu_probe import probe_gpu
from app.services.training.registry import get_available_providers, get_provider

router = APIRouter(prefix="/training", tags=["training"])


@router.get("/providers", response_model=TrainingProvidersRead)
def list_training_providers() -> TrainingProvidersRead:
    gpu = probe_gpu()
    return TrainingProvidersRead(
        available=get_available_providers(),
        gpu=GPUInfoRead(
            torch_version=gpu.torch_version,
            cuda_available=gpu.cuda_available,
            device_name=gpu.device_name,
            vram_total_mb=gpu.vram_total_mb,
            cuda_version=gpu.cuda_version,
        ),
    )


@router.post("/jobs", response_model=TrainingJobRead, status_code=status.HTTP_202_ACCEPTED)
def create_training_job(payload: TrainingJobCreate, db: Session = Depends(get_db)) -> TrainingJob:
    version = db.get(DatasetVersion, payload.dataset_version_id)
    if version is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset version not found")

    from app.models.dataset import Dataset

    dataset = db.get(Dataset, version.dataset_id)

    try:
        provider = get_provider(payload.provider.value)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    job = TrainingJob(
        project_id=dataset.project_id,
        dataset_version_id=payload.dataset_version_id,
        base_model_id=payload.base_model_id,
        provider=payload.provider,
        status=TrainingJobStatus.QUEUED,
        epochs=payload.epochs,
        batch_size=payload.batch_size,
        image_size=payload.image_size,
        learning_rate=payload.learning_rate,
        device=payload.device,
        extra_args=payload.extra_args,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    provider.start_training(db, job)
    db.refresh(job)
    return job


@router.get("/jobs", response_model=list[TrainingJobRead])
def list_training_jobs(project_id: uuid.UUID, db: Session = Depends(get_db)) -> list[TrainingJob]:
    return list(
        db.scalars(select(TrainingJob).where(TrainingJob.project_id == project_id).order_by(TrainingJob.created_at.desc()))
    )


@router.get("/jobs/{job_id}", response_model=TrainingJobRead)
def get_training_job(job_id: uuid.UUID, db: Session = Depends(get_db)) -> TrainingJob:
    job = db.get(TrainingJob, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Training job not found")
    return job


@router.get("/jobs/{job_id}/epochs", response_model=list[TrainingJobEpochRead])
def list_training_job_epochs(job_id: uuid.UUID, db: Session = Depends(get_db)) -> list[TrainingJobEpoch]:
    job = db.get(TrainingJob, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Training job not found")
    return list(
        db.scalars(
            select(TrainingJobEpoch).where(TrainingJobEpoch.training_job_id == job_id).order_by(TrainingJobEpoch.epoch)
        )
    )


@router.post("/jobs/{job_id}/cancel", response_model=TrainingJobRead)
def cancel_training_job(job_id: uuid.UUID, db: Session = Depends(get_db)) -> TrainingJob:
    job = db.get(TrainingJob, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Training job not found")
    provider = get_provider(job.provider.value)
    provider.cancel_training(db, job)
    db.refresh(job)
    return job
