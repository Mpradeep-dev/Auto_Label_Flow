"""Training jobs, exercised through the real API with a fake Ultralytics
YOLO trainer (real GPU training is validated once, manually, against the
RTX 5060 — not something the automated suite can require). Uses
`real_client`/`real_db_session` since the task opens its own DB session,
same reasoning as the video/inference-batch tests."""
from __future__ import annotations

import io
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage
from sqlalchemy.orm import Session

from app.services.inference import registry
from app.workers.progress import request_cancel


class _FakeDetectionModel:
    def __init__(self, weights_path: str) -> None:
        pass

    @property
    def class_names(self) -> dict[int, str]:
        return {0: "ball", 1: "cone", 2: "cone_1"}

    def predict(self, image, conf=0.20, iou=0.70, imgsz=640) -> list:
        return []


class _FakeTrainer:
    def __init__(self, epoch: int) -> None:
        self.epoch = epoch
        self.metrics = {
            "metrics/precision(B)": 0.5,
            "metrics/recall(B)": 0.4,
            "metrics/mAP50(B)": 0.3,
            "metrics/mAP50-95(B)": 0.2,
        }
        self.loss_names = ["box_loss", "cls_loss", "dfl_loss"]
        self.tloss = [0.10, 0.20, 0.30]
        self.stop = False


class _FakeYOLO:
    last_instance: "_FakeYOLO | None" = None

    def __init__(self, weights_path: str) -> None:
        self.weights_path = weights_path
        self._callback = None
        _FakeYOLO.last_instance = self

    def add_callback(self, event: str, fn) -> None:
        if event == "on_fit_epoch_end":
            self._callback = fn

    def train(self, **kwargs) -> None:
        self.last_kwargs = kwargs
        for e in range(kwargs["epochs"]):
            trainer = _FakeTrainer(e)
            if self._callback:
                self._callback(trainer)
            if trainer.stop:
                break
        weights_dir = Path(kwargs["project"]) / kwargs["name"] / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        (weights_dir / "best.pt").write_bytes(b"fake-trained-weights")


@pytest.fixture(autouse=True)
def _patch_training(monkeypatch):
    registry.clear_cache()
    monkeypatch.setattr(registry, "UltralyticsDetectionModel", _FakeDetectionModel)
    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", _FakeYOLO)
    yield
    registry.clear_cache()


def _jpeg_bytes() -> bytes:
    img = PILImage.new("RGB", (64, 48), color=(80, 80, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def version_and_base_model(real_client: TestClient, unique_name: str, tmp_path) -> tuple[str, str, str]:
    project = real_client.post("/api/v1/projects", json={"name": unique_name}).json()
    real_client.patch(
        f"/api/v1/projects/{project['id']}",
        json={"class_config": [{"id": 0, "name": "ball"}, {"id": 1, "name": "cone"}, {"id": 2, "name": "cone_1"}]},
    )
    dataset = real_client.post(f"/api/v1/projects/{project['id']}/datasets", json={"name": "d"}).json()

    image = real_client.post(
        f"/api/v1/datasets/{dataset['id']}/images", files={"file": ("f.jpg", _jpeg_bytes(), "image/jpeg")}
    ).json()
    real_client.post(f"/api/v1/images/{image['id']}/approve")
    real_client.post(
        "/api/v1/annotations",
        json={"image_id": image["id"], "class_id": 1, "class_name": "cone", "x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.3},
    )
    version = real_client.post(f"/api/v1/datasets/{dataset['id']}/versions", json={}).json()

    weights = tmp_path / "detect_v1.pt"
    base_model = real_client.post(
        "/api/v1/models", json={"name": "detect_v1", "weights_path": str(weights), "kind": "DETECTOR"}
    ).json()

    return project["id"], version["id"], base_model["id"]


def test_training_job_completes_and_registers_new_model(
    real_client: TestClient, version_and_base_model
) -> None:
    project_id, version_id, base_model_id = version_and_base_model

    resp = real_client.post(
        "/api/v1/training/jobs",
        json={"dataset_version_id": version_id, "base_model_id": base_model_id, "epochs": 3, "batch_size": 2},
    )
    assert resp.status_code == 202, resp.text
    job = resp.json()
    assert job["status"] == "COMPLETED"
    assert job["current_epoch"] == 3
    assert job["result_model_id"] is not None
    assert job["started_at"] is not None
    assert job["completed_at"] is not None

    epochs = real_client.get(f"/api/v1/training/jobs/{job['id']}/epochs").json()
    assert len(epochs) == 3
    assert [e["epoch"] for e in epochs] == [1, 2, 3]
    assert epochs[-1]["map50"] == pytest.approx(0.3)

    new_model = real_client.get(f"/api/v1/models/{job['result_model_id']}").json()
    assert new_model["base_model_id"] == base_model_id
    assert new_model["class_config"] == [
        {"id": 0, "name": "ball"},
        {"id": 1, "name": "cone"},
        {"id": 2, "name": "cone_1"},
    ]

    jobs_for_project = real_client.get(f"/api/v1/training/jobs?project_id={project_id}").json()
    assert any(j["id"] == job["id"] for j in jobs_for_project)


def test_training_job_extra_args_reach_yolo_train_and_typed_fields_win(
    real_client: TestClient, version_and_base_model
) -> None:
    """`extra_args` should reach the real `YOLO.train()` call as extra
    kwargs, but must never override the job's own typed fields — a
    `patience` the user typed in should show up, an `epochs` they also
    (accidentally or not) put in `extra_args` must not beat the job's real
    `epochs=3`."""
    _, version_id, base_model_id = version_and_base_model

    resp = real_client.post(
        "/api/v1/training/jobs",
        json={
            "dataset_version_id": version_id,
            "base_model_id": base_model_id,
            "epochs": 3,
            "batch_size": 2,
            "extra_args": {"patience": 20, "optimizer": "AdamW", "dropout": 0.1, "epochs": 999},
        },
    )
    assert resp.status_code == 202, resp.text
    job = resp.json()
    assert job["status"] == "COMPLETED"
    assert job["extra_args"] == {"patience": 20, "optimizer": "AdamW", "dropout": 0.1, "epochs": 999}

    used_kwargs = _FakeYOLO.last_instance.last_kwargs
    assert used_kwargs["patience"] == 20
    assert used_kwargs["optimizer"] == "AdamW"
    assert used_kwargs["dropout"] == 0.1
    assert used_kwargs["epochs"] == 3  # the job's real epochs, not extra_args' 999
    assert job["current_epoch"] == 3  # confirms it actually trained for 3, not 999


def test_training_job_missing_version_404(real_client: TestClient, version_and_base_model) -> None:
    _, _, base_model_id = version_and_base_model
    resp = real_client.post(
        "/api/v1/training/jobs",
        json={"dataset_version_id": str(uuid.uuid4()), "base_model_id": base_model_id, "epochs": 1},
    )
    assert resp.status_code == 404


def test_training_job_kaggle_unavailable_400(real_client: TestClient, version_and_base_model) -> None:
    _, version_id, base_model_id = version_and_base_model
    resp = real_client.post(
        "/api/v1/training/jobs",
        json={
            "dataset_version_id": version_id,
            "base_model_id": base_model_id,
            "provider": "KAGGLE",
            "epochs": 1,
        },
    )
    assert resp.status_code == 400


def test_training_providers_endpoint_reports_local_only(real_client: TestClient) -> None:
    """No Kaggle credentials are set in the test environment — this IS the
    path every install without Kaggle takes (PLAN 'the unconfigured path is
    the default path, and it will be the one that gets tested')."""
    resp = real_client.get("/api/v1/training/providers")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] == ["LOCAL"]
    assert isinstance(body["gpu"]["cuda_available"], bool)
    assert body["gpu"]["torch_version"]


def test_cancel_flag_checked_at_epoch_boundary_stops_early(
    real_client: TestClient, real_db_session: Session, version_and_base_model
) -> None:
    """Sets the cancel flag before the task runs and confirms the
    on_fit_epoch_end callback's cancel-check stops training after the
    first epoch instead of running all 5 — proving the check is real, not
    just plumbing that's never exercised."""
    from app.models.dataset import Dataset
    from app.models.dataset_version import DatasetVersion
    from app.models.training_job import TrainingJob, TrainingJobStatus
    from app.workers.tasks.training import train_local_model

    _, version_id, base_model_id = version_and_base_model

    version_row = real_db_session.get(DatasetVersion, uuid.UUID(version_id))
    dataset = real_db_session.get(Dataset, version_row.dataset_id)

    job = TrainingJob(
        project_id=dataset.project_id,
        dataset_version_id=version_row.id,
        base_model_id=uuid.UUID(base_model_id),
        provider="LOCAL",
        status=TrainingJobStatus.QUEUED,
        epochs=5,
        batch_size=2,
    )
    real_db_session.add(job)
    real_db_session.commit()
    real_db_session.refresh(job)

    request_cancel(str(job.id))
    train_local_model(str(job.id))  # direct call, not .delay() — same eager-equivalent pattern as inference's test

    real_db_session.refresh(job)
    assert job.status == TrainingJobStatus.CANCELLED
    assert job.current_epoch == 1  # stopped after the first epoch's boundary check, not all 5
    assert job.result_model_id is None  # cancelled runs must not register a model


def test_deleting_project_cascades_through_a_completed_training_job(
    real_client: TestClient, version_and_base_model
) -> None:
    """Regression: deleting a project whose dataset version has a training
    job on it used to 500 with a ForeignKeyViolation — `training_jobs
    .dataset_version_id` was ON DELETE RESTRICT, which blocked the
    project -> dataset -> dataset_version cascade this endpoint promises
    ('permanently deletes everything inside it', per the frontend's
    delete-confirmation copy). See training_job.py for the fix."""
    project_id, version_id, base_model_id = version_and_base_model

    job_resp = real_client.post(
        "/api/v1/training/jobs",
        json={"dataset_version_id": version_id, "base_model_id": base_model_id, "epochs": 1, "batch_size": 2},
    )
    assert job_resp.status_code == 202, job_resp.text
    assert job_resp.json()["status"] == "COMPLETED"

    resp = real_client.delete(f"/api/v1/projects/{project_id}")
    assert resp.status_code == 204, resp.text
    assert real_client.get(f"/api/v1/projects/{project_id}").status_code == 404
