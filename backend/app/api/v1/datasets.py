from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.security import stream_upload_to_temp
from app.db.session import get_db
from app.models.dataset import Dataset
from app.models.image import Image, ImageReviewStatus
from app.models.project import Project
from app.models.roboflow_job import RoboflowJob, RoboflowJobKind
from app.models.video import Video
from app.schemas.dashboard import DatasetStatistics, ErrorAnalysis
from app.schemas.dataset import DatasetCreate, DatasetRead, DatasetStats
from app.schemas.integration import RoboflowImportRequest, RoboflowJobRead
from app.services.dataset.error_analysis import compute_error_analysis
from app.services.dataset.import_coco import CocoImportError, import_coco_zip
from app.services.dataset.import_cvat import CvatImportError, import_cvat_zip
from app.services.dataset.statistics import compute_dataset_statistics
from app.services.integrations.roboflow_connect import RoboflowNotConnectedError, get_client
from app.workers.tasks.roboflow import run_roboflow_import

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


@router.post(
    "/projects/{project_id}/import/roboflow", response_model=RoboflowJobRead, status_code=status.HTTP_202_ACCEPTED
)
def import_dataset_from_roboflow(
    project_id: uuid.UUID, payload: RoboflowImportRequest, db: Session = Depends(get_db)
) -> RoboflowJob:
    """Dispatches a background job and returns immediately (PLAN-follow-on:
    a whole-project pull can be thousands of images, too long to hold the
    request open for) — poll `GET /integrations/roboflow/jobs/{id}` or
    follow `/stream` for progress, same pattern as `inference_jobs.py`.
    Still checks the connection synchronously first: failing fast on "not
    connected" beats creating a job that's doomed before it starts."""
    _get_project_or_404(project_id, db)
    try:
        get_client(db)
    except RoboflowNotConnectedError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    job = RoboflowJob(
        project_id=project_id,
        kind=RoboflowJobKind.IMPORT,
        workspace=payload.workspace,
        project_slug=payload.project,
        version=payload.version,
        dataset_name=payload.dataset_name,
        unannotated_only=payload.unannotated_only,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    run_roboflow_import.delay(str(job.id))
    db.refresh(job)  # eager/test mode: already terminal by the time we return
    return job


@router.post("/projects/{project_id}/import/coco", response_model=DatasetRead, status_code=status.HTTP_201_CREATED)
async def import_dataset_from_coco(
    project_id: uuid.UUID,
    file: UploadFile,
    dataset_name: str | None = Form(default=None),
    db: Session = Depends(get_db),
) -> Dataset:
    """Runs synchronously (unlike the Roboflow import) — this is a zip
    already sitting on disk, not a network pull from an external API, so
    there's no per-image round trip to make a background job worth the
    extra machinery for at this app's scale."""
    _get_project_or_404(project_id, db)
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Expected a .zip file")

    tmp_path = await stream_upload_to_temp(file, ".zip")
    try:
        return import_coco_zip(db, project_id=project_id, zip_path=tmp_path, dataset_name=dataset_name)
    except CocoImportError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/projects/{project_id}/import/cvat", response_model=DatasetRead, status_code=status.HTTP_201_CREATED)
async def import_dataset_from_cvat(
    project_id: uuid.UUID,
    file: UploadFile,
    dataset_name: str | None = Form(default=None),
    db: Session = Depends(get_db),
) -> Dataset:
    _get_project_or_404(project_id, db)
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Expected a .zip file")

    tmp_path = await stream_upload_to_temp(file, ".zip")
    try:
        return import_cvat_zip(db, project_id=project_id, zip_path=tmp_path, dataset_name=dataset_name)
    except CvatImportError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


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
