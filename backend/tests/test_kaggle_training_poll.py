"""Kaggle training: the cancel-doesn't-actually-cancel bug, and the
poll task that's the real fix for a Kaggle job sitting frozen at
RUNNING/epoch 0 forever (see workers/tasks/kaggle_training.py's own
docstring for why nothing ever updated one before this existed).

Uses a fake Kaggle API object (house convention — no real Kaggle
credentials in this environment, same reasoning kaggle_provider.py's own
module docstring gives) and the same fake-YOLO monkeypatch
test_training_jobs.py uses, since a completed-job poll registers a new
model, which probes the (fake) weights the same way any other
registration does.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.models.training_job import TrainingJob, TrainingJobStatus, TrainingProviderType
from app.services.inference import registry
from app.services.training.kaggle_provider import KaggleTrainingProvider
from app.workers.tasks.kaggle_training import poll_kaggle_training_jobs


class _FakeDetectionModel:
    def __init__(self, weights_path: str) -> None:
        pass

    @property
    def class_names(self) -> dict[int, str]:
        return {0: "ball", 1: "cone"}

    def predict(self, image, conf=0.20, iou=0.70, imgsz=640) -> list:
        return []


@pytest.fixture(autouse=True)
def _patch_registry(monkeypatch):
    registry.clear_cache()
    monkeypatch.setattr(registry, "UltralyticsDetectionModel", _FakeDetectionModel)
    yield
    registry.clear_cache()


class _FakeKaggleApi:
    """Stands in for kaggle.api.kaggle_api_extended.KaggleApi — only the
    three methods KaggleTrainingProvider actually calls post-push."""

    def __init__(self, status: str) -> None:
        self.status = status
        self.output_calls: list[tuple[str, str]] = []

    def kernels_status(self, ref: str):
        return {"status": self.status}

    def kernels_output(self, ref: str, path: str, quiet: bool = True) -> None:
        self.output_calls.append((ref, path))
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "best.pt").write_bytes(b"fake-kaggle-trained-weights")
        (out_dir / f"{ref.split('/')[-1]}.log").write_text("epoch 1/3 ...\ndone", encoding="utf-8")


def _make_kaggle_job(db, *, project_id: str, version_id: str, base_model_id: str, status=TrainingJobStatus.RUNNING) -> TrainingJob:
    job_id = uuid.uuid4()
    job = TrainingJob(
        id=job_id,
        project_id=uuid.UUID(project_id),
        dataset_version_id=uuid.UUID(version_id),
        base_model_id=uuid.UUID(base_model_id),
        provider=TrainingProviderType.KAGGLE,
        status=status,
        epochs=3,
        # Unique per job (mirrors kaggle_provider.py's own
        # `annotate-train-{job.id[:8]}` slug) — `real_db_session` commits
        # for real across this whole test-session database, so several of
        # these tests' jobs coexist; a shared ref would make them
        # indistinguishable from each other via `_FakeKaggleApi.output_calls`.
        kaggle_kernel_ref=f"testuser/annotate-train-{str(job_id)[:8]}",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def test_cancel_sets_status_cancelled_with_honest_note(real_db_session, version_and_base_model) -> None:
    """The actual bug report: clicking Cancel on a Kaggle job visibly did
    nothing, because `cancel_training` never moved `status` off RUNNING —
    it only appended a note to `error`. Exercised at the provider level (not
    through POST /training/jobs/{id}/cancel) since that endpoint itself
    gates on Kaggle being configured, which this test environment isn't —
    same reasoning test_training_job_kaggle_unavailable_400 documents."""
    job = _make_kaggle_job(real_db_session, project_id=version_and_base_model[0], version_id=version_and_base_model[1], base_model_id=version_and_base_model[2])

    KaggleTrainingProvider().cancel_training(real_db_session, job)

    real_db_session.refresh(job)
    assert job.status == TrainingJobStatus.CANCELLED
    assert job.failed_at is not None
    assert "no remote-stop" in job.error.lower()


def test_poll_completed_job_downloads_and_registers_model(
    monkeypatch, real_db_session, version_and_base_model
) -> None:
    project_id, version_id, base_model_id = version_and_base_model
    job = _make_kaggle_job(real_db_session, project_id=project_id, version_id=version_id, base_model_id=base_model_id)
    fake_api = _FakeKaggleApi(status="complete")
    monkeypatch.setattr(KaggleTrainingProvider, "_client", lambda self: fake_api)

    counts = poll_kaggle_training_jobs()

    assert counts["polled"] == 1
    assert counts["completed"] == 1
    real_db_session.refresh(job)
    assert job.status == TrainingJobStatus.COMPLETED
    assert job.result_model_id is not None
    assert job.artifact_path and Path(job.artifact_path).exists()
    assert job.completed_at is not None

    from app.models.ml_model import MLModel

    result_model = real_db_session.get(MLModel, job.result_model_id)
    assert result_model.base_model_id == uuid.UUID(base_model_id)
    assert result_model.class_config == [{"id": 0, "name": "ball"}, {"id": 1, "name": "cone"}]


def test_poll_failed_job_records_kaggle_logs(monkeypatch, real_db_session, version_and_base_model) -> None:
    project_id, version_id, base_model_id = version_and_base_model
    job = _make_kaggle_job(real_db_session, project_id=project_id, version_id=version_id, base_model_id=base_model_id)
    fake_api = _FakeKaggleApi(status="error")
    monkeypatch.setattr(KaggleTrainingProvider, "_client", lambda self: fake_api)

    counts = poll_kaggle_training_jobs()

    assert counts["failed"] == 1
    real_db_session.refresh(job)
    assert job.status == TrainingJobStatus.FAILED
    assert job.failed_at is not None
    assert "epoch 1/3" in job.error  # pulled from the fake kernel log


def test_poll_heartbeats_updated_at_when_still_running(
    monkeypatch, real_db_session, version_and_base_model
) -> None:
    project_id, version_id, base_model_id = version_and_base_model
    job = _make_kaggle_job(real_db_session, project_id=project_id, version_id=version_id, base_model_id=base_model_id)
    backdated = datetime.now(timezone.utc) - timedelta(hours=1)
    real_db_session.query(TrainingJob).filter(TrainingJob.id == job.id).update({"updated_at": backdated})
    real_db_session.commit()

    fake_api = _FakeKaggleApi(status="running")  # unchanged from job.status — status itself won't move
    monkeypatch.setattr(KaggleTrainingProvider, "_client", lambda self: fake_api)

    poll_kaggle_training_jobs()

    real_db_session.refresh(job)
    assert job.status == TrainingJobStatus.RUNNING
    # The heartbeat, not a status change, is what should have moved —
    # otherwise a long-but-healthy Kaggle run looks stale to
    # tasks/reconcile.py purely because polling itself never touched it.
    assert job.updated_at > backdated + timedelta(minutes=30)


def test_poll_ignores_terminal_jobs(monkeypatch, real_db_session, version_and_base_model) -> None:
    """A COMPLETED job must not be re-polled or re-finalized — the query
    only selects QUEUED/RUNNING. (Doesn't assert a global `polled` count:
    `real_db_session` genuinely commits, so other tests' Kaggle jobs in
    this same run can still be sitting in the shared test DB in
    QUEUED/RUNNING — this only asserts THIS job was left alone.)"""
    project_id, version_id, base_model_id = version_and_base_model
    completed = _make_kaggle_job(
        real_db_session, project_id=project_id, version_id=version_id, base_model_id=base_model_id,
        status=TrainingJobStatus.COMPLETED,
    )
    fake_api = _FakeKaggleApi(status="complete")
    monkeypatch.setattr(KaggleTrainingProvider, "_client", lambda self: fake_api)

    poll_kaggle_training_jobs()

    real_db_session.refresh(completed)
    assert completed.status == TrainingJobStatus.COMPLETED
    assert completed.result_model_id is None  # never touched
    # Never asked Kaggle about THIS job specifically (other tests' leftover
    # RUNNING jobs in this same shared, really-committing DB may still get
    # legitimately polled here — that's correct behavior, not noise).
    assert all(ref != completed.kaggle_kernel_ref for ref, _ in fake_api.output_calls)
