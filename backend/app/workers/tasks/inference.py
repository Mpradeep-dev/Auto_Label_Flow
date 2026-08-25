"""Batch auto-annotation — runs on the `gpu` queue (concurrency=1). Reuses
exactly the predict -> filter -> persist sequence the synchronous
per-image endpoint uses (`api/v1/annotations.py::auto_annotate_image`); the
only things a background job adds are progress reporting and
cancellation.
"""
from __future__ import annotations

import uuid

import cv2
import numpy as np
from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.annotation import AnnotationSource
from app.models.image import Image
from app.models.inference_job import InferenceJob, JobStatus
from app.services.annotation import service as annotation_service
from app.services.inference.registry import get_detection_model
from app.services.quality.filters import FilterConfig, filter_predictions
from app.services.storage.factory import get_storage
from app.workers.celery_app import celery_app
from app.workers.progress import ThrottledProgressWriter, clear_cancel, is_cancel_requested

_DB_CHECKPOINT_EVERY = 5  # commit the durable job row every N images, not every single one


@celery_app.task(bind=True, name="app.workers.tasks.inference.run_inference_batch")
def run_inference_batch(self, job_id: str) -> None:
    db = SessionLocal()
    job = db.get(InferenceJob, uuid.UUID(job_id))
    if job is None:
        db.close()
        return

    job.status = JobStatus.RUNNING
    job.celery_task_id = self.request.id
    db.commit()

    try:
        images = list(db.scalars(select(Image).where(Image.dataset_id == job.dataset_id)))
        job.total_images = len(images)
        db.commit()

        writer = ThrottledProgressWriter(job_id, len(images))
        detector = get_detection_model(db, job.model_id)  # loaded once, reused for the whole batch
        storage = get_storage()

        for i, image in enumerate(images):
            if is_cancel_requested(job_id):
                job.status = JobStatus.CANCELLED
                db.commit()
                writer.finish(status="CANCELLED")
                return

            predictions_made = 0
            try:
                data = storage.read_bytes(image.storage_key)
                arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if arr is None:
                    raise ValueError("stored image could not be decoded")

                raw = detector.predict(arr, conf=job.conf, iou=job.iou)
                filtered = filter_predictions(raw, FilterConfig())

                for existing in annotation_service.list_annotations_for_image(db, image.id):
                    if existing.source == AnnotationSource.AUTO:
                        annotation_service.delete_annotation(db, annotation_id=existing.id)

                created = annotation_service.bulk_create_from_predictions(
                    db,
                    image_id=image.id,
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
                predictions_made = len(created)
                job.processed_images += 1
                job.total_predictions += predictions_made
            except Exception as exc:  # one bad image must not abort the whole batch
                job.failed_images += 1
                job.error = f"image {image.id}: {exc}"

            if i % _DB_CHECKPOINT_EVERY == 0 or i == len(images) - 1:
                db.commit()
            writer.update(i + 1, predictions_delta=predictions_made)

        job.status = JobStatus.COMPLETED
        db.commit()
        writer.finish()
    except Exception as exc:
        db.rollback()
        job = db.get(InferenceJob, uuid.UUID(job_id))
        if job is not None:
            job.status = JobStatus.FAILED
            job.error = str(exc)
            db.commit()
        raise
    finally:
        clear_cancel(job_id)
        db.close()
