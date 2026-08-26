"""Regression tests for audit finding REL-01's reconciliation sweep
(app/workers/tasks/reconcile.py) — the safety net for a job left stuck at
QUEUED/RUNNING by a worker that crashed or was killed mid-task, where the
task's own exception handling never got a chance to run at all."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.models.inference_job import InferenceJob, JobStatus
from app.models.ml_model import MLModel, ModelKind
from app.models.video import Video, VideoStatus
from app.workers.celery_app import RECONCILE_STALE_AFTER_S
from app.workers.tasks.reconcile import reconcile_stale_jobs


def _make_model(db, name: str) -> MLModel:
    # Inserted directly via the ORM, not the registration endpoint — a
    # RESTRICT FK means InferenceJob.model_id must reference a real row,
    # but this test doesn't need loadable weights, just a valid id.
    model = MLModel(name=name, version="v1", kind=ModelKind.DETECTOR, weights_path="/nonexistent/x.pt")
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


def test_reconcile_fails_a_stale_running_inference_job(real_client: TestClient, real_db_session, unique_name: str) -> None:
    project = real_client.post("/api/v1/projects", json={"name": unique_name}).json()
    dataset = real_client.post(f"/api/v1/projects/{project['id']}/datasets", json={"name": "d"}).json()
    model = _make_model(real_db_session, f"detect-{uuid.uuid4().hex[:8]}")
    job = InferenceJob(
        project_id=uuid.UUID(project["id"]),
        dataset_id=uuid.UUID(dataset["id"]),
        model_id=model.id,
        status=JobStatus.RUNNING,
    )
    real_db_session.add(job)
    real_db_session.commit()
    # Simulate a job whose worker died long ago: backdate updated_at past
    # the inference staleness threshold directly (bypassing the ORM's
    # onupdate, which would otherwise reset it to "now").
    stale_at = datetime.now(timezone.utc) - timedelta(seconds=RECONCILE_STALE_AFTER_S["inference"] + 60)
    real_db_session.query(InferenceJob).filter(InferenceJob.id == job.id).update({"updated_at": stale_at})
    real_db_session.commit()

    reconcile_stale_jobs()

    real_db_session.refresh(job)
    assert job.status == JobStatus.FAILED
    assert job.error


def test_reconcile_leaves_a_recently_updated_running_job_alone(real_client: TestClient, real_db_session, unique_name: str) -> None:
    project = real_client.post("/api/v1/projects", json={"name": unique_name}).json()
    dataset = real_client.post(f"/api/v1/projects/{project['id']}/datasets", json={"name": "d"}).json()
    model = _make_model(real_db_session, f"detect-{uuid.uuid4().hex[:8]}")
    job = InferenceJob(
        project_id=uuid.UUID(project["id"]),
        dataset_id=uuid.UUID(dataset["id"]),
        model_id=model.id,
        status=JobStatus.RUNNING,
    )
    real_db_session.add(job)
    real_db_session.commit()

    reconcile_stale_jobs()

    real_db_session.refresh(job)
    assert job.status == JobStatus.RUNNING


def test_reconcile_fails_a_stale_extracting_video(real_client: TestClient, real_db_session, unique_name: str) -> None:
    project = real_client.post("/api/v1/projects", json={"name": unique_name}).json()
    dataset = real_client.post(f"/api/v1/projects/{project['id']}/datasets", json={"name": "d"}).json()
    video = Video(
        project_id=uuid.UUID(project["id"]),
        dataset_id=uuid.UUID(dataset["id"]),
        storage_key="videos/fake.mp4",
        original_filename="fake.mp4",
        status=VideoStatus.EXTRACTING,
    )
    real_db_session.add(video)
    real_db_session.commit()
    stale_at = datetime.now(timezone.utc) - timedelta(seconds=RECONCILE_STALE_AFTER_S["video"] + 60)
    real_db_session.query(Video).filter(Video.id == video.id).update({"updated_at": stale_at})
    real_db_session.commit()

    reconcile_stale_jobs()

    real_db_session.refresh(video)
    assert video.status == VideoStatus.FAILED
    assert video.error
