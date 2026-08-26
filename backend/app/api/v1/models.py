from __future__ import annotations

import uuid
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import stream_upload_to_temp
from app.db.session import get_db
from app.models.image import Image
from app.models.ml_model import MLModel, ModelKind
from app.schemas.dashboard import ModelMetricsUpdate
from app.schemas.model import ModelDownloadRequest, ModelRead, ModelRegisterRequest, ModelUpdateRequest
from app.schemas.prediction import PredictResponse, PredictionOut
from app.services.inference.detector import ModelLoadError
from app.services.inference.registry import (
    ModelInUseError,
    delete_model,
    get_detection_model,
    get_project_class_names,
    register_model,
    register_model_from_upload,
    register_model_from_url,
    rename_model,
)
from app.services.quality.filters import FilterConfig, filter_predictions
from app.services.storage.factory import get_storage

router = APIRouter(prefix="/models", tags=["models"])

_ALLOWED_WEIGHTS_EXTENSIONS = (".pt",)


@router.post("", response_model=ModelRead, status_code=status.HTTP_201_CREATED)
def create_model(payload: ModelRegisterRequest, db: Session = Depends(get_db)) -> MLModel:
    try:
        return register_model(
            db,
            name=payload.name,
            weights_path=payload.weights_path,
            kind=payload.kind,
            version=payload.version,
            framework=payload.framework,
        )
    except ModelLoadError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


@router.post("/download", response_model=ModelRead, status_code=status.HTTP_201_CREATED)
def create_model_from_url(payload: ModelDownloadRequest, db: Session = Depends(get_db)) -> MLModel:
    """Same as create_model, but fetches the weights from `url` into
    ARTIFACTS_DIR first — lets the frontend register a model by pasting a
    download link instead of the weights already sitting on this host."""
    try:
        return register_model_from_url(
            db,
            name=payload.name,
            url=payload.url,
            kind=payload.kind,
            version=payload.version,
            framework=payload.framework,
        )
    except ModelLoadError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


@router.post("/upload", response_model=ModelRead, status_code=status.HTTP_201_CREATED)
async def create_model_from_upload(
    file: UploadFile,
    name: str = Form(...),
    kind: ModelKind = Form(...),
    version: str = Form("v1"),
    framework: str = Form("ultralytics"),
    db: Session = Depends(get_db),
) -> MLModel:
    """Browser-upload counterpart to create_model_from_url: lets the
    frontend send a weights file straight from the user's machine (picked
    via a native file dialog) instead of requiring it already be reachable
    by path or URL from inside this container."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _ALLOWED_WEIGHTS_EXTENSIONS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Unsupported weights extension {ext!r}. Allowed: {', '.join(_ALLOWED_WEIGHTS_EXTENSIONS)}",
        )
    temp_path = await stream_upload_to_temp(file, ext)
    try:
        return register_model_from_upload(
            db, name=name, temp_path=temp_path, suffix=ext, kind=kind, version=version, framework=framework
        )
    except ModelLoadError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


@router.get("", response_model=list[ModelRead])
def list_models(db: Session = Depends(get_db)) -> list[MLModel]:
    return list(db.scalars(select(MLModel).order_by(MLModel.created_at.desc())))


@router.get("/{model_id}", response_model=ModelRead)
def get_model(model_id: uuid.UUID, db: Session = Depends(get_db)) -> MLModel:
    model = db.get(MLModel, model_id)
    if model is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found")
    return model


@router.patch("/{model_id}", response_model=ModelRead)
def update_model(model_id: uuid.UUID, payload: ModelUpdateRequest, db: Session = Depends(get_db)) -> MLModel:
    model = db.get(MLModel, model_id)
    if model is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found")
    return rename_model(db, model, payload.name)


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_model(model_id: uuid.UUID, db: Session = Depends(get_db)) -> None:
    model = db.get(MLModel, model_id)
    if model is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found")
    try:
        delete_model(db, model)
    except ModelInUseError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


@router.put("/{model_id}/metrics", response_model=ModelRead)
def update_model_metrics(model_id: uuid.UUID, payload: ModelMetricsUpdate, db: Session = Depends(get_db)) -> MLModel:
    """PLAN spec section 14: 'Allow uploading evaluation metrics for every
    model version.' Replaces the stored metrics dict wholesale — the
    caller is expected to send the full evaluation result (mAP50,
    mAP50-95, precision, recall, fps, latency, model_size, per_class:
    {class_name: {precision, recall}}), not a partial patch."""
    model = db.get(MLModel, model_id)
    if model is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found")
    model.metrics = payload.metrics
    db.commit()
    db.refresh(model)
    return model


@router.post("/{model_id}/predict", response_model=PredictResponse)
def predict(
    model_id: uuid.UUID,
    image_id: uuid.UUID = Query(..., description="An already-uploaded image to run detection on"),
    conf: float = Query(0.20, ge=0.0, le=1.0),
    iou: float = Query(0.70, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
) -> PredictResponse:
    """Synchronous single-image predict — the Phase 2 validation path.
    Batch inference over a whole dataset runs as a Celery job (Phase 4) and
    reuses this exact model-cache + filtering pipeline, just fed from a
    frame-extraction loop instead of one HTTP request."""
    image = db.get(Image, image_id)
    if image is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image not found")

    try:
        class_names = get_project_class_names(db, image.project_id)
        detector = get_detection_model(db, model_id, class_names=class_names)
    except ModelLoadError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    image_bytes = get_storage().read_bytes(image.storage_key)
    import numpy as np

    arr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Stored image could not be decoded")

    raw = detector.predict(arr, conf=conf, iou=iou)
    filtered = filter_predictions(raw, FilterConfig())

    return PredictResponse(
        model_id=str(model_id),
        image_width=image.width,
        image_height=image.height,
        raw_count=len(raw),
        filtered_count=len(filtered),
        predictions=[
            PredictionOut(
                class_id=d.class_id,
                class_name=d.class_name,
                confidence=d.confidence,
                x1=d.x1,
                y1=d.y1,
                x2=d.x2,
                y2=d.y2,
            )
            for d in filtered
        ],
    )
