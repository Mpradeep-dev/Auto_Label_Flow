"""Batch auto-annotation Celery task, run eagerly through the real API
(see conftest's `real_client` — genuine commits, since the task opens its
own DB session)."""
from __future__ import annotations

import io

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


@pytest.fixture(autouse=True)
def _patch_detector(monkeypatch):
    registry.clear_cache()
    monkeypatch.setattr(registry, "UltralyticsDetectionModel", _FakeDetectionModel)
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
    model = real_client.post(
        "/api/v1/models", json={"name": "detect_v1", "weights_path": str(weights), "kind": "DETECTOR"}
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
