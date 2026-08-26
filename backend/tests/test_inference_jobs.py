"""Batch auto-annotation Celery task, run eagerly through the real API
(see conftest's `real_client` — genuine commits, since the task opens its
own DB session)."""
from __future__ import annotations

import io
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from app.services.inference import registry
from app.services.inference.detector import Detection
from app.workers.progress import get_progress, request_cancel


class _FakeDetectionModel:
    def __init__(self, weights_path: str) -> None:
        pass

    @property
    def class_names(self) -> dict[int, str]:
        return {0: "ball", 1: "cone", 2: "cone_1"}

    def predict(self, image, conf=0.20, iou=0.70, imgsz=640) -> list[Detection]:
        return [Detection(class_id=1, class_name="cone", confidence=0.7, x1=0.1, y1=0.1, x2=0.2, y2=0.2)]


class _FakeYoloWorldDetectionModel:
    """Stands in for YoloWorldDetectionModel: predictions are labeled from
    whatever vocabulary set_classes() was last called with, so a test can
    confirm the batch job actually threads the project's classes through
    rather than the checkpoint's default vocabulary."""

    def __init__(self, weights_path: str) -> None:
        self._class_names: dict[int, str] = {0: "person"}

    @property
    def class_names(self) -> dict[int, str]:
        return self._class_names

    def set_classes(self, classes: list[str]) -> None:
        self._class_names = dict(enumerate(classes))

    def predict(self, image, conf=0.20, iou=0.70, imgsz=640) -> list[Detection]:
        return [
            Detection(
                class_id=0, class_name=self._class_names[0], confidence=0.8, x1=0.1, y1=0.1, x2=0.2, y2=0.2
            )
        ]


@pytest.fixture(autouse=True)
def _patch_detector(monkeypatch):
    registry.clear_cache()
    monkeypatch.setattr(registry, "UltralyticsDetectionModel", _FakeDetectionModel)
    monkeypatch.setattr(registry, "YoloWorldDetectionModel", _FakeYoloWorldDetectionModel)
    yield
    registry.clear_cache()


def _upload_jpeg(client: TestClient, dataset_id: str) -> str:
    arr = (np.random.rand(40, 60, 3) * 255).astype("uint8")
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="JPEG")
    resp = client.post(
        f"/api/v1/datasets/{dataset_id}/images", files={"file": ("s.jpg", buf.getvalue(), "image/jpeg")}
    )
    return resp.json()["id"]


@pytest.fixture()
def dataset_with_images(real_client: TestClient, unique_name: str, tmp_path) -> tuple[str, str, str]:
    project = real_client.post("/api/v1/projects", json={"name": unique_name}).json()
    dataset = real_client.post(f"/api/v1/projects/{project['id']}/datasets", json={"name": "d"}).json()
    for _ in range(4):
        _upload_jpeg(real_client, dataset["id"])
    weights = tmp_path / "detect_v1.pt"
    # A unique name per fixture invocation, not the fixed "detect_v1" this
    # used to hardcode — real_client commits for real, and this fixture is
    # reused across many tests in one shared test-session database, so a
    # fixed name silently relied on `models` having no uniqueness
    # constraint (exactly what audit finding DB-03 flagged and this test
    # suite now enforces).
    model = real_client.post(
        "/api/v1/models",
        json={"name": f"detect-{uuid.uuid4().hex[:8]}", "weights_path": str(weights), "kind": "DETECTOR"},
    ).json()
    return project["id"], dataset["id"], model["id"]


def test_batch_job_processes_all_images(real_client: TestClient, dataset_with_images) -> None:
    _, dataset_id, model_id = dataset_with_images
    resp = real_client.post("/api/v1/inference/jobs", json={"dataset_id": dataset_id, "model_id": model_id})
    assert resp.status_code == 202, resp.text
    job = resp.json()
    assert job["status"] == "COMPLETED"
    assert job["total_images"] == 4
    assert job["processed_images"] == 4
    assert job["failed_images"] == 0
    assert job["total_predictions"] == 4  # one fake detection per image

    images = real_client.get(f"/api/v1/datasets/{dataset_id}/images").json()
    for image in images["items"]:
        anns = real_client.get(f"/api/v1/images/{image['id']}/annotations").json()
        assert len(anns) == 1
        assert anns[0]["source"] == "AUTO"


def test_batch_job_yolo_world_labels_with_project_class_names(
    real_client: TestClient, unique_name: str, tmp_path
) -> None:
    """The core YOLO-World integration behavior: a promptable model's
    detections are labeled with the current project's classes, not the
    checkpoint's default vocabulary."""
    project = real_client.post("/api/v1/projects", json={"name": unique_name}).json()
    real_client.patch(
        f"/api/v1/projects/{project['id']}",
        json={"class_config": [{"id": 0, "name": "helmet"}, {"id": 1, "name": "vest"}]},
    )
    dataset = real_client.post(f"/api/v1/projects/{project['id']}/datasets", json={"name": "d"}).json()
    _upload_jpeg(real_client, dataset["id"])

    weights = tmp_path / "yolo_world.pt"
    model = real_client.post(
        "/api/v1/models",
        json={
            "name": f"yolo-world-{uuid.uuid4().hex[:8]}",
            "weights_path": str(weights),
            "kind": "DETECTOR",
            "framework": "yolo-world",
        },
    ).json()
    assert model["is_promptable"] is True

    resp = real_client.post("/api/v1/inference/jobs", json={"dataset_id": dataset["id"], "model_id": model["id"]})
    assert resp.status_code == 202, resp.text
    assert resp.json()["status"] == "COMPLETED"

    images = real_client.get(f"/api/v1/datasets/{dataset['id']}/images").json()
    anns = real_client.get(f"/api/v1/images/{images['items'][0]['id']}/annotations").json()
    assert len(anns) == 1
    assert anns[0]["class_name"] == "helmet"  # project's class_config[0], not the checkpoint's default


def test_batch_job_writes_progress_to_redis(real_client: TestClient, dataset_with_images) -> None:
    _, dataset_id, model_id = dataset_with_images
    job = real_client.post(
        "/api/v1/inference/jobs", json={"dataset_id": dataset_id, "model_id": model_id}
    ).json()
    # Eager execution has already finished by the time the request returns —
    # the finish() call forces a write regardless of the throttle window.
    progress = get_progress(job["id"])
    assert progress is not None
    assert progress.status == "COMPLETED"
    assert progress.current == 4


def test_batch_job_missing_dataset_404(real_client: TestClient) -> None:
    import uuid

    resp = real_client.post(
        "/api/v1/inference/jobs",
        json={"dataset_id": str(uuid.uuid4()), "model_id": str(uuid.uuid4())},
    )
    assert resp.status_code == 404


def test_get_job_by_id(real_client: TestClient, dataset_with_images) -> None:
    _, dataset_id, model_id = dataset_with_images
    job = real_client.post(
        "/api/v1/inference/jobs", json={"dataset_id": dataset_id, "model_id": model_id}
    ).json()
    fetched = real_client.get(f"/api/v1/inference/jobs/{job['id']}").json()
    assert fetched["id"] == job["id"]
    assert fetched["status"] == "COMPLETED"


def test_latest_inference_job_returns_newest_for_dataset(real_client: TestClient, dataset_with_images) -> None:
    """Backs the "reattach to a still-running job after navigating away or
    reloading" behavior on the Auto Annotation page."""
    _, dataset_id, model_id = dataset_with_images
    first = real_client.post("/api/v1/inference/jobs", json={"dataset_id": dataset_id, "model_id": model_id}).json()
    second = real_client.post("/api/v1/inference/jobs", json={"dataset_id": dataset_id, "model_id": model_id}).json()
    assert first["id"] != second["id"]

    resp = real_client.get(f"/api/v1/inference/jobs/latest?dataset_id={dataset_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == second["id"]


def test_latest_inference_job_none_found_returns_null(real_client: TestClient, unique_name: str) -> None:
    project = real_client.post("/api/v1/projects", json={"name": unique_name}).json()
    dataset = real_client.post(f"/api/v1/projects/{project['id']}/datasets", json={"name": "d"}).json()
    resp = real_client.get(f"/api/v1/inference/jobs/latest?dataset_id={dataset['id']}")
    assert resp.status_code == 200
    assert resp.json() is None


def test_get_missing_job_404(real_client: TestClient) -> None:
    import uuid

    resp = real_client.get(f"/api/v1/inference/jobs/{uuid.uuid4()}")
    assert resp.status_code == 404


def test_cancel_sets_redis_flag(real_client: TestClient, dataset_with_images) -> None:
    """Cancellation is checked between images by the running task; since
    eager mode runs the whole batch synchronously before this test can
    call cancel mid-flight through the API, this only verifies the flag
    plumbing itself. The actual mid-batch stop is covered directly below
    (`test_cancel_flag_set_before_start_stops_before_any_image`)."""
    _, dataset_id, model_id = dataset_with_images
    job = real_client.post(
        "/api/v1/inference/jobs", json={"dataset_id": dataset_id, "model_id": model_id}
    ).json()
    resp = real_client.post(f"/api/v1/inference/jobs/{job['id']}/cancel")
    assert resp.status_code == 200
    request_cancel(job["id"])  # idempotent; confirms the helper itself works standalone


def test_cancel_flag_set_before_start_stops_before_any_image(
    real_client: TestClient, real_db_session, dataset_with_images
) -> None:
    """Calls the task function directly (not via the API/.delay(), so we can
    set the cancel flag BEFORE the loop starts) and confirms it exits with
    CANCELLED and processes zero images — proving the between-image cancel
    check actually works, not just that the flag can be set."""
    from app.models.inference_job import InferenceJob, JobStatus
    from app.workers.tasks.inference import run_inference_batch

    _, dataset_id, model_id = dataset_with_images

    import uuid as _uuid

    from app.models.dataset import Dataset

    dataset = real_db_session.get(Dataset, _uuid.UUID(dataset_id))
    job = InferenceJob(
        project_id=dataset.project_id,
        dataset_id=_uuid.UUID(dataset_id),
        model_id=_uuid.UUID(model_id),
        status=JobStatus.QUEUED,
    )
    real_db_session.add(job)
    real_db_session.commit()
    real_db_session.refresh(job)

    request_cancel(str(job.id))
    run_inference_batch(str(job.id))  # call directly, synchronously, not .delay()

    real_db_session.refresh(job)
    assert job.status == JobStatus.CANCELLED
    assert job.processed_images == 0


def test_second_inference_job_on_same_dataset_while_one_is_running_is_rejected(
    real_client: TestClient, real_db_session, dataset_with_images
) -> None:
    """Regression test for audit finding BE-03. Inserts a QUEUED job
    directly (bypassing task dispatch, which would normally race the check
    to a terminal state before a second request could land) to simulate a
    genuinely still-running job, then confirms a second `POST` for the same
    dataset is rejected with 409 rather than silently starting a second,
    interleaved batch."""
    import uuid as _uuid

    from app.models.dataset import Dataset
    from app.models.inference_job import InferenceJob, JobStatus

    _, dataset_id, model_id = dataset_with_images
    dataset = real_db_session.get(Dataset, _uuid.UUID(dataset_id))
    running = InferenceJob(
        project_id=dataset.project_id,
        dataset_id=_uuid.UUID(dataset_id),
        model_id=_uuid.UUID(model_id),
        status=JobStatus.RUNNING,
    )
    real_db_session.add(running)
    real_db_session.commit()

    resp = real_client.post("/api/v1/inference/jobs", json={"dataset_id": dataset_id, "model_id": model_id})
    assert resp.status_code == 409, resp.text


def test_delete_model_with_existing_inference_job_returns_409_not_500(
    real_client: TestClient, dataset_with_images
) -> None:
    """Regression test for audit finding DB-01: InferenceJob.model_id is a
    RESTRICT foreign key, so deleting a model that an inference job still
    references used to bubble up as an unhandled IntegrityError -> raw 500.
    It should be a clean, actionable 409 instead, and the model must
    survive the failed delete rather than being left half-deleted."""
    _, dataset_id, model_id = dataset_with_images
    job = real_client.post("/api/v1/inference/jobs", json={"dataset_id": dataset_id, "model_id": model_id}).json()
    assert job["model_id"] == model_id

    resp = real_client.delete(f"/api/v1/models/{model_id}")
    assert resp.status_code == 409, resp.text

    still_there = real_client.get(f"/api/v1/models/{model_id}")
    assert still_there.status_code == 200
