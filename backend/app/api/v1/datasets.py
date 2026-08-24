from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.dataset import Dataset
from app.models.image import Image, ImageReviewStatus
from app.models.project import Project
from app.models.video import Video
from app.schemas.dashboard import DatasetStatistics, ErrorAnalysis
from app.schemas.dataset import DatasetCreate, DatasetRead, DatasetStats
from app.services.dataset.error_analysis import compute_error_analysis
from app.services.dataset.statistics import compute_dataset_statistics

router = APIRouter(tags=["datasets"])


def _get_project_or_404(project_id: uuid.UUID, db: Session) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    return project


@router.post("/projects/{project_id}/datasets", response_model=DatasetRead, status_code=status.HTTP_201_CREATED)
def create_dataset(project_id: uuid.UUID, payload: DatasetCreate, db: Session = Depends(get_db)) -> Dataset:
    _get_project_or_404(project_id, db)
    dataset = Dataset(project_id=project_id, name=payload.name, description=payload.description)
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


@router.get("/projects/{project_id}/datasets", response_model=list[DatasetRead])
def list_datasets(project_id: uuid.UUID, db: Session = Depends(get_db)) -> list[Dataset]:
    _get_project_or_404(project_id, db)
    return list(
        db.scalars(select(Dataset).where(Dataset.project_id == project_id).order_by(Dataset.created_at.desc()))
    )


@router.get("/datasets/{dataset_id}", response_model=DatasetRead)
def get_dataset(dataset_id: uuid.UUID, db: Session = Depends(get_db)) -> Dataset:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found")
    return dataset


@router.get("/datasets/{dataset_id}/stats", response_model=DatasetStats)
def get_dataset_stats(dataset_id: uuid.UUID, db: Session = Depends(get_db)) -> DatasetStats:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found")

    total_images = db.scalar(select(func.count()).select_from(Image).where(Image.dataset_id == dataset_id)) or 0
    approved = (
        db.scalar(
            select(func.count())
            .select_from(Image)
            .where(Image.dataset_id == dataset_id, Image.review_status == ImageReviewStatus.APPROVED)
        )
        or 0
    )
    pending = (
        db.scalar(
            select(func.count())
            .select_from(Image)
            .where(Image.dataset_id == dataset_id, Image.review_status == ImageReviewStatus.PENDING)
        )
        or 0
    )
    total_videos = db.scalar(select(func.count()).select_from(Video).where(Video.dataset_id == dataset_id)) or 0

    return DatasetStats(
        total_images=total_images,
        pending_images=pending,
        approved_images=approved,
        total_videos=total_videos,
    )


@router.get("/datasets/{dataset_id}/statistics", response_model=DatasetStatistics)
def get_dataset_statistics(dataset_id: uuid.UUID, db: Session = Depends(get_db)) -> DatasetStatistics:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found")
    return DatasetStatistics(**compute_dataset_statistics(db, dataset_id))


@router.get("/datasets/{dataset_id}/error-analysis", response_model=ErrorAnalysis)
def get_error_analysis(dataset_id: uuid.UUID, db: Session = Depends(get_db)) -> ErrorAnalysis:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found")
    return ErrorAnalysis(**compute_error_analysis(db, dataset_id))


@router.delete("/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(dataset_id: uuid.UUID, db: Session = Depends(get_db)) -> None:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found")
    db.delete(dataset)
    db.commit()
