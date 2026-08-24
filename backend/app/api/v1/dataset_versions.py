from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.dataset import Dataset
from app.models.dataset_version import DatasetVersion, DatasetVersionStatus
from app.schemas.dataset_version import DatasetVersionCreate, DatasetVersionRead
from app.services.dataset.export_yolo import ExportError, export_yolo
from app.services.dataset.versioning import NoApprovedImagesError, create_version
from app.services.storage.factory import get_storage

router = APIRouter(tags=["dataset-versions"])


def _to_read(version: DatasetVersion) -> DatasetVersionRead:
    download_url = get_storage().get_url(version.export_storage_key) if version.export_storage_key else None
    return DatasetVersionRead(
        id=version.id,
        dataset_id=version.dataset_id,
        version_number=version.version_number,
        status=version.status,
        train_ratio=version.train_ratio,
        val_ratio=version.val_ratio,
        test_ratio=version.test_ratio,
        split_seed=version.split_seed,
        used_frame_level_fallback=version.used_frame_level_fallback,
        total_images=version.total_images,
        total_annotations=version.total_annotations,
        error=version.error,
        download_url=download_url,
        created_at=version.created_at,
    )


@router.post(
    "/datasets/{dataset_id}/versions", response_model=DatasetVersionRead, status_code=status.HTTP_201_CREATED
)
def create_dataset_version(
    dataset_id: uuid.UUID, payload: DatasetVersionCreate, db: Session = Depends(get_db)
) -> DatasetVersionRead:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found")
    try:
        version = create_version(
            db,
            dataset_id=dataset_id,
            train_ratio=payload.train_ratio,
            val_ratio=payload.val_ratio,
            test_ratio=payload.test_ratio,
            seed=payload.seed,
        )
    except NoApprovedImagesError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return _to_read(version)


@router.get("/datasets/{dataset_id}/versions", response_model=list[DatasetVersionRead])
def list_dataset_versions(dataset_id: uuid.UUID, db: Session = Depends(get_db)) -> list[DatasetVersionRead]:
    rows = list(
        db.scalars(
            select(DatasetVersion)
            .where(DatasetVersion.dataset_id == dataset_id)
            .order_by(DatasetVersion.version_number.desc())
        )
    )
    return [_to_read(v) for v in rows]


@router.get("/versions/{version_id}", response_model=DatasetVersionRead)
def get_dataset_version(version_id: uuid.UUID, db: Session = Depends(get_db)) -> DatasetVersionRead:
    version = db.get(DatasetVersion, version_id)
    if version is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset version not found")
    return _to_read(version)


@router.post("/versions/{version_id}/export", response_model=DatasetVersionRead)
def export_dataset_version(version_id: uuid.UUID, db: Session = Depends(get_db)) -> DatasetVersionRead:
    version = db.get(DatasetVersion, version_id)
    if version is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset version not found")

    version.status = DatasetVersionStatus.EXPORTING
    db.commit()
    try:
        key = export_yolo(db, version_id=version_id)
    except ExportError as exc:
        version.status = DatasetVersionStatus.FAILED
        version.error = str(exc)
        db.commit()
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except Exception as exc:
        version.status = DatasetVersionStatus.FAILED
        version.error = str(exc)
        db.commit()
        raise

    version.status = DatasetVersionStatus.EXPORTED
    version.export_storage_key = key
    db.commit()
    db.refresh(version)
    return _to_read(version)
