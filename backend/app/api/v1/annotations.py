from __future__ import annotations

import uuid

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.v1.images import _to_read as _image_to_read
from app.db.session import get_db
from app.models.annotation import Annotation, AnnotationSource
from app.models.image import Image
from app.schemas.annotation import (
    AnnotationCreate,
    AnnotationDelete,
    AnnotationRead,
    AnnotationUpdate,
    AutoAnnotateRequest,
)
from app.schemas.image import ImageRead
from app.services.annotation import service as annotation_service
from app.services.inference.detector import ModelLoadError
from app.services.inference.registry import get_detection_model, get_project_class_names
from app.services.quality.filters import FilterConfig, filter_predictions
from app.services.storage.factory import get_storage

router = APIRouter(tags=["annotations"])


def _get_image_or_404(image_id: uuid.UUID, db: Session) -> Image:
    image = db.get(Image, image_id)
    if image is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image not found")
    return image


@router.get("/images/{image_id}/annotations", response_model=list[AnnotationRead])
def list_annotations(image_id: uuid.UUID, db: Session = Depends(get_db)) -> list[Annotation]:
    _get_image_or_404(image_id, db)
    return annotation_service.list_annotations_for_image(db, image_id)


@router.post("/annotations", response_model=AnnotationRead, status_code=status.HTTP_201_CREATED)
def create_annotation(payload: AnnotationCreate, db: Session = Depends(get_db)) -> Annotation:
    _get_image_or_404(payload.image_id, db)
    # Geometry-shape validation (bbox ordering, polygon point count/range)
    # already ran in AnnotationCreate's model_validator.
    try:
        return annotation_service.create_annotation(
            db,
            image_id=payload.image_id,
            class_id=payload.class_id,
            class_name=payload.class_name,
            shape_type=payload.shape_type,
            points=[list(p) for p in payload.points] if payload.points else None,
            x1=payload.x1,
            y1=payload.y1,
            x2=payload.x2,
            y2=payload.y2,
            confidence=payload.confidence,
            source=AnnotationSource.HUMAN,
        )
    except annotation_service.InvalidGeometryError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


@router.put("/annotations/{annotation_id}", response_model=AnnotationRead)
def update_annotation(
    annotation_id: uuid.UUID, payload: AnnotationUpdate, db: Session = Depends(get_db)
) -> Annotation:
    # Bbox-ordering / polygon-point-count validation already ran in
    # AnnotationUpdate's model_validator; whether x1..y2 vs. points is
    # actually valid for THIS annotation's shape_type is enforced by the
    # service layer (it's the only place that knows the existing shape_type).
    try:
        return annotation_service.update_annotation(
            db,
            annotation_id=annotation_id,
            class_id=payload.class_id,
            class_name=payload.class_name,
            x1=payload.x1,
            y1=payload.y1,
            x2=payload.x2,
            y2=payload.y2,
            points=[list(p) for p in payload.points] if payload.points else None,
            confidence=payload.confidence,
        )
    except annotation_service.AnnotationNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Annotation not found")
    except annotation_service.InvalidGeometryError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


@router.delete("/annotations/{annotation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_annotation(
    annotation_id: uuid.UUID, payload: AnnotationDelete | None = None, db: Session = Depends(get_db)
) -> None:
    payload = payload or AnnotationDelete()
    try:
        annotation_service.delete_annotation(
            db,
            annotation_id=annotation_id,
            error_category=payload.error_category,
            error_reason=payload.error_reason,
        )
    except annotation_service.AnnotationNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Annotation not found")


@router.post("/annotations/{annotation_id}/duplicate", response_model=AnnotationRead, status_code=status.HTTP_201_CREATED)
def duplicate_annotation(annotation_id: uuid.UUID, db: Session = Depends(get_db)) -> Annotation:
    try:
        return annotation_service.duplicate_annotation(db, annotation_id=annotation_id)
    except annotation_service.AnnotationNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Annotation not found")


@router.post("/images/{image_id}/approve", response_model=ImageRead)
def approve_image(image_id: uuid.UUID, db: Session = Depends(get_db)) -> ImageRead:
    try:
        image = annotation_service.approve_image(db, image_id=image_id)
    except annotation_service.AnnotationNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image not found")
    return _image_to_read(image)


@router.post("/images/{image_id}/reject", response_model=ImageRead)
def reject_image(image_id: uuid.UUID, db: Session = Depends(get_db)) -> ImageRead:
    try:
        image = annotation_service.reject_image(db, image_id=image_id)
    except annotation_service.AnnotationNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image not found")
    return _image_to_read(image)


@router.post("/images/{image_id}/auto-annotate", response_model=list[AnnotationRead])
def auto_annotate_image(
    image_id: uuid.UUID, payload: AutoAnnotateRequest, db: Session = Depends(get_db)
) -> list[Annotation]:
    """Runs the registered detector on this image and persists the results
    as AUTO annotations — the fusion of Phase 2's synchronous predict path
    with Phase 3's storage layer. Batch auto-annotation over a whole
    dataset (Phase 4) reuses this same predict -> filter -> persist
    sequence per image, just driven by a Celery task instead of one
    request."""
    image = _get_image_or_404(image_id, db)

    try:
        class_names = get_project_class_names(db, image.project_id)
        detector = get_detection_model(db, payload.model_id, class_names=class_names)
    except ModelLoadError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    image_bytes = get_storage().read_bytes(image.storage_key)
    arr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Stored image could not be decoded")

    raw = detector.predict(arr, conf=payload.conf, iou=payload.iou)
    filtered = filter_predictions(raw, FilterConfig())

    if payload.replace_existing:
        for existing in annotation_service.list_annotations_for_image(db, image_id):
            if existing.source == AnnotationSource.AUTO:
                annotation_service.delete_annotation(db, annotation_id=existing.id)

    return annotation_service.bulk_create_from_predictions(
        db,
        image_id=image_id,
        project_id=image.project_id,
        predictions=[
            {
                "class_id": d.class_id,
                "class_name": d.class_name,
                "confidence": d.confidence,
                "x1": d.x1,
                "y1": d.y1,
                "x2": d.x2,
                "y2": d.y2,
            }
            for d in filtered
        ],
    )
