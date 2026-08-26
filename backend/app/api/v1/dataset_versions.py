from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.dataset import Dataset
from app.models.dataset_version import DatasetVersion, DatasetVersionStatus
from app.models.roboflow_job import RoboflowJob, RoboflowJobKind
from app.schemas.dataset_version import DatasetVersionCreate, DatasetVersionRead
from app.schemas.integration import RoboflowExportRequest, RoboflowJobRead
from app.services.dataset.export_coco import ExportError as CocoExportError, export_coco
from app.services.dataset.export_cvat import ExportError as CvatExportError, export_cvat
from app.services.dataset.export_yolo import ExportError, export_yolo
from app.services.dataset.versioning import NoApprovedImagesError, VersionNumberConflictError, create_version
from app.services.integrations.roboflow_connect import RoboflowNotConnectedError, get_client
from app.services.storage.factory import get_storage
from app.workers.tasks.roboflow import run_roboflow_export

router = APIRouter(tags=["dataset-versions"])


def _to_read(version: DatasetVersion) -> DatasetVersionRead:
    storage = get_storage()
    download_url = storage.get_url(version.export_storage_key) if version.export_storage_key else None
    coco_download_url = storage.get_url(version.coco_export_storage_key) if version.coco_export_storage_key else None
    cvat_download_url = storage.get_url(version.cvat_export_storage_key) if version.cvat_export_storage_key else None
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
        coco_download_url=coco_download_url,
        cvat_download_url=cvat_download_url,
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
    except VersionNumberConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
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


@router.post("/versions/{version_id}/export/coco", response_model=DatasetVersionRead)
def export_dataset_version_coco(version_id: uuid.UUID, db: Session = Depends(get_db)) -> DatasetVersionRead:
    """Doesn't touch `version.status` (that's the YOLO export's terminal
    state machine) — this and the CVAT-XML export are independent,
    additional artifacts a version can carry, not another status to be in."""
    version = db.get(DatasetVersion, version_id)
    if version is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset version not found")
    try:
        version.coco_export_storage_key = export_coco(db, version_id=version_id)
    except CocoExportError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    db.commit()
    db.refresh(version)
    return _to_read(version)


@router.post("/versions/{version_id}/export/cvat", response_model=DatasetVersionRead)
def export_dataset_version_cvat(version_id: uuid.UUID, db: Session = Depends(get_db)) -> DatasetVersionRead:
    version = db.get(DatasetVersion, version_id)
    if version is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset version not found")
    try:
        version.cvat_export_storage_key = export_cvat(db, version_id=version_id)
    except CvatExportError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    db.commit()
    db.refresh(version)
    return _to_read(version)


@router.post(
    "/versions/{version_id}/export/roboflow", response_model=RoboflowJobRead, status_code=status.HTTP_202_ACCEPTED
)
def export_dataset_version_to_roboflow(
    version_id: uuid.UUID, payload: RoboflowExportRequest, db: Session = Depends(get_db)
) -> RoboflowJob:
    """Dispatches a background job rather than pushing synchronously — see
    `import_dataset_from_roboflow` in `datasets.py` for the same reasoning
    and the same fail-fast-on-"not connected" check."""
    version = db.get(DatasetVersion, version_id)
    if version is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset version not found")
    dataset = db.get(Dataset, version.dataset_id)

    try:
        get_client(db)
    except RoboflowNotConnectedError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    job = RoboflowJob(
        project_id=dataset.project_id,
        kind=RoboflowJobKind.EXPORT,
        workspace=payload.workspace,
        project_slug=payload.project,
        dataset_version_id=version_id,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    run_roboflow_export.delay(str(job.id))
    db.refresh(job)  # eager/test mode: already terminal by the time we return
    return job
