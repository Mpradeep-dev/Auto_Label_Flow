"""Roboflow import/export — runs on the `default` queue (network-bound
upload/download, not GPU work). Wraps the synchronous service functions in
`services/integrations/roboflow_{import,export}.py` with the same
DB-checkpoint + Redis-progress pattern `tasks/inference.py` uses, so both
directions get a pollable job row and an SSE-able progress bar instead of
tying up the request for however long a whole-project push/pull takes.
"""
from __future__ import annotations

import uuid

from app.db.session import SessionLocal
from app.models.roboflow_job import RoboflowJob, RoboflowJobStatus
from app.services.integrations.roboflow_export import RoboflowExportError, push_version_to_roboflow
from app.services.integrations.roboflow_import import import_roboflow_project, import_roboflow_raw_project
from app.workers.celery_app import ROBOFLOW_SOFT_TIME_LIMIT_S, ROBOFLOW_TIME_LIMIT_S, celery_app
from app.workers.progress import ThrottledProgressWriter, clear_cancel, is_cancel_requested

_DB_CHECKPOINT_EVERY = 5  # commit the durable job row every N items, not every single one
_MAX_ERROR_LEN = 1900  # RoboflowJob.error is String(2000); leave headroom


def _make_progress_cb(job: RoboflowJob, db, writer: ThrottledProgressWriter):
    def cb(current: int, total: int, failed: int = 0) -> None:
        if writer.total != total:
            writer.total = total
            job.total_items = total
        # `current` is real successes only (export: images that reached
        # Roboflow); `failed` is attempts that didn't. `processed_items`
        # stays a success count so a stalled/failing job never looks like
        # it's making headway.
        job.processed_items = current
        job.failed_count = failed
        done = current + failed
        edge = done == 0 or total == 0 or done == total
        if edge or done % _DB_CHECKPOINT_EVERY == 0:
            db.commit()
        writer.update(current, force=edge, failed=failed)

    return cb


@celery_app.task(
    bind=True,
    name="app.workers.tasks.roboflow.run_roboflow_import",
    time_limit=ROBOFLOW_TIME_LIMIT_S,
    soft_time_limit=ROBOFLOW_SOFT_TIME_LIMIT_S,
)
def run_roboflow_import(self, job_id: str) -> None:
    db = SessionLocal()
    job = db.get(RoboflowJob, uuid.UUID(job_id))
    if job is None:
        db.close()
        return

    job.status = RoboflowJobStatus.RUNNING
    job.celery_task_id = self.request.id
    db.commit()

    writer = ThrottledProgressWriter(job_id, total=0)

    def should_cancel() -> bool:
        return is_cancel_requested(job_id)

    try:
        # `job.version is None` means the project has no generated Version
        # to pull — fall back to reading its raw uploaded images directly
        # (see `import_roboflow_raw_project`'s docstring). Whichever path
        # ran, the rest of this task treats it identically.
        if job.version is None:
            dataset = import_roboflow_raw_project(
                db,
                project_id=job.project_id,
                workspace=job.workspace,
                project_slug=job.project_slug,
                dataset_name=job.dataset_name,
                unannotated_only=job.unannotated_only,
                batch_id=job.batch_id,
                progress_cb=_make_progress_cb(job, db, writer),
                should_cancel=should_cancel,
            )
        else:
            dataset = import_roboflow_project(
                db,
                project_id=job.project_id,
                workspace=job.workspace,
                project_slug=job.project_slug,
                version=job.version,
                dataset_name=job.dataset_name,
                progress_cb=_make_progress_cb(job, db, writer),
                should_cancel=should_cancel,
            )
        job.result_dataset_id = dataset.id
        if is_cancel_requested(job_id):
            job.status = RoboflowJobStatus.CANCELLED
            db.commit()
            writer.finish(status="CANCELLED")
        else:
            job.status = RoboflowJobStatus.COMPLETED
            db.commit()
            writer.finish()
    except Exception as exc:
        db.rollback()
        job = db.get(RoboflowJob, uuid.UUID(job_id))
        if job is not None:
            job.status = RoboflowJobStatus.FAILED
            job.error = str(exc)
            db.commit()
        writer.finish(status="FAILED", error=str(exc))
        raise
    finally:
        clear_cancel(job_id)
        db.close()


@celery_app.task(
    bind=True,
    name="app.workers.tasks.roboflow.run_roboflow_export",
    time_limit=ROBOFLOW_TIME_LIMIT_S,
    soft_time_limit=ROBOFLOW_SOFT_TIME_LIMIT_S,
)
def run_roboflow_export(self, job_id: str) -> None:
    db = SessionLocal()
    job = db.get(RoboflowJob, uuid.UUID(job_id))
    if job is None:
        db.close()
        return

    job.status = RoboflowJobStatus.RUNNING
    job.celery_task_id = self.request.id
    db.commit()

    writer = ThrottledProgressWriter(job_id, total=0)
    try:
        uploaded, failed, failures = push_version_to_roboflow(
            db,
            version_id=job.dataset_version_id,
            workspace=job.workspace,
            project_slug=job.project_slug,
            progress_cb=_make_progress_cb(job, db, writer),
            should_cancel=lambda: is_cancel_requested(job_id),
        )
        job.uploaded_count = uploaded
        job.failed_count = failed
        job.failures = failures
        if is_cancel_requested(job_id):
            job.status = RoboflowJobStatus.CANCELLED
            db.commit()
            writer.finish(status="CANCELLED")
        elif uploaded == 0 and failed > 0:
            # Every image failed, but the version was smaller than the
            # service's fail-fast threshold so it returned instead of
            # raising. Still a failed export, not a COMPLETED one that
            # silently pushed nothing.
            job.status = RoboflowJobStatus.FAILED
            job.error = (
                f"All {failed} image(s) failed to upload to Roboflow — nothing was pushed. "
                f"First error: {failures[0]}"
                if failures
                else f"All {failed} image(s) failed to upload to Roboflow — nothing was pushed."
            )[:_MAX_ERROR_LEN]
            db.commit()
            writer.finish(status="FAILED", error=job.error)
        else:
            job.status = RoboflowJobStatus.COMPLETED
            db.commit()
            writer.finish()
    except RoboflowExportError as exc:
        # Expected terminal failure (quota / plan / key / Roboflow down) —
        # the message is already written for the user. Record it and stop;
        # this is not a crash to propagate to the caller.
        db.rollback()
        job = db.get(RoboflowJob, uuid.UUID(job_id))
        if job is not None:
            job.status = RoboflowJobStatus.FAILED
            job.error = str(exc)[:_MAX_ERROR_LEN]
            db.commit()
        writer.finish(status="FAILED", error=str(exc))
    except Exception as exc:
        db.rollback()
        job = db.get(RoboflowJob, uuid.UUID(job_id))
        if job is not None:
            job.status = RoboflowJobStatus.FAILED
            job.error = str(exc)
            db.commit()
        writer.finish(status="FAILED", error=str(exc))
        raise
    finally:
        clear_cancel(job_id)
        db.close()
