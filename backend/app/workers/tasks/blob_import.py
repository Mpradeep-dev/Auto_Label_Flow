"""Azure-Blob import as a background job — the same DB-checkpoint +
Redis-progress wrapper `tasks/roboflow.py` puts around a synchronous
service call, so the import (which walks a whole container prefix) gets a
pollable job row and an SSE-able progress bar instead of holding the
request open. Runs on the `default` queue (network/IO-bound, not GPU).
"""
from __future__ import annotations

import uuid

from app.db.session import SessionLocal
from app.models.blob_import_job import BlobImportJob, BlobImportJobStatus
from app.services.integrations.azure_blob_import import import_azure_blob_prefix
from app.workers.celery_app import BLOB_IMPORT_SOFT_TIME_LIMIT_S, BLOB_IMPORT_TIME_LIMIT_S, celery_app
from app.workers.progress import ThrottledProgressWriter, clear_cancel, is_cancel_requested

_DB_CHECKPOINT_EVERY = 5  # commit the durable job row every N items, not every single one


def _make_progress_cb(job: BlobImportJob, db, writer: ThrottledProgressWriter):
    def cb(current: int, total: int) -> None:
        if writer.total != total:
            writer.total = total
            job.total_items = total
        job.processed_items = current
        edge = current == 0 or total == 0 or current == total
        if edge or current % _DB_CHECKPOINT_EVERY == 0:
            db.commit()
        writer.update(current, force=edge)

    return cb


@celery_app.task(
    bind=True,
    name="app.workers.tasks.blob_import.run_blob_import",
    time_limit=BLOB_IMPORT_TIME_LIMIT_S,
    soft_time_limit=BLOB_IMPORT_SOFT_TIME_LIMIT_S,
)
def run_blob_import(self, job_id: str) -> None:
    db = SessionLocal()
    job = db.get(BlobImportJob, uuid.UUID(job_id))
    if job is None:
        db.close()
        return

    job.status = BlobImportJobStatus.RUNNING
    job.celery_task_id = self.request.id
    db.commit()

    writer = ThrottledProgressWriter(job_id, total=0)

    try:
        dataset = import_azure_blob_prefix(
            db,
            project_id=job.project_id,
            prefix=job.prefix,
            label_format=job.label_format,
            dataset_name=job.dataset_name,
            progress_cb=_make_progress_cb(job, db, writer),
            should_cancel=lambda: is_cancel_requested(job_id),
        )
        job.result_dataset_id = dataset.id
        if is_cancel_requested(job_id):
            job.status = BlobImportJobStatus.CANCELLED
            db.commit()
            writer.finish(status="CANCELLED")
        else:
            job.status = BlobImportJobStatus.COMPLETED
            db.commit()
            writer.finish()
    except Exception as exc:
        db.rollback()
        job = db.get(BlobImportJob, uuid.UUID(job_id))
        if job is not None:
            job.status = BlobImportJobStatus.FAILED
            job.error = str(exc)
            db.commit()
        writer.finish(status="FAILED", error=str(exc))
        raise
    finally:
        clear_cancel(job_id)
        db.close()
