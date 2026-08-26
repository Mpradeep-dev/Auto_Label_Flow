from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from app.services.inference import registry
from app.services.inference.detector import Detection


class _FakeDetectionModel:
    def __init__(self, weights_path: str) -> None:
        self._weights_path = weights_path

    @property
    def class_names(self) -> dict[int, str]:
        return {0: "ball", 1: "cone", 2: "cone_1"}

    def predict(self, image, conf=0.20, iou=0.70, imgsz=640) -> list[Detection]:
        return [
            Detection(class_id=1, class_name="cone", confidence=0.72, x1=0.20, y1=0.58, x2=0.23, y2=0.64),
            # A near-identical duplicate of the same box — exercises the
            # filtering layer's dedup through the full HTTP round trip.
            Detection(class_id=1, class_name="cone", confidence=0.30, x1=0.201, y1=0.581, x2=0.231, y2=0.641),
            Detection(class_id=0, class_name="ball", confidence=0.85, x1=0.39, y1=0.60, x2=0.42, y2=0.66),
        ]


class _FakeYoloWorldDetectionModel:
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
                class_id=0, class_name=self._class_names[0], confidence=0.9, x1=0.1, y1=0.1, x2=0.2, y2=0.2
            )
        ]


@pytest.fixture(autouse=True)
def _patch_detector(monkeypatch):
    registry.clear_cache()
    monkeypatch.setattr(registry, "UltralyticsDetectionModel", _FakeDetectionModel)
    monkeypatch.setattr(registry, "YoloWorldDetectionModel", _FakeYoloWorldDetectionModel)
    yield
    registry.clear_cache()


def _upload_test_image(client: TestClient, dataset_id: str) -> str:
    arr = (np.random.rand(60, 80, 3) * 255).astype("uint8")
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="JPEG")
    resp = client.post(
        f"/api/v1/datasets/{dataset_id}/images",
        files={"file": ("sample.jpg", buf.getvalue(), "image/jpeg")},
    )
    assert resp.status_code == 201
    return resp.json()["id"]


def test_register_model_and_list(client: TestClient, tmp_path: Path) -> None:
    weights = tmp_path / "detect_v1.pt"
    resp = client.post(
        "/api/v1/models",
        json={"name": "detect_v1", "weights_path": str(weights), "kind": "DETECTOR"},
    )
    assert resp.status_code == 201, resp.text
    model = resp.json()
    assert model["class_config"] == [
        {"id": 0, "name": "ball"},
        {"id": 1, "name": "cone"},
        {"id": 2, "name": "cone_1"},
    ]

    listed = client.get("/api/v1/models").json()
    assert any(m["id"] == model["id"] for m in listed)


def test_predict_endpoint_returns_filtered_detections(client: TestClient, tmp_path: Path, unique_name: str) -> None:
    weights = tmp_path / "detect_v1.pt"
    model = client.post(
        "/api/v1/models",
        json={"name": "detect_v1", "weights_path": str(weights), "kind": "DETECTOR"},
    ).json()

    project = client.post("/api/v1/projects", json={"name": unique_name}).json()
    dataset = client.post(f"/api/v1/projects/{project['id']}/datasets", json={"name": "d"}).json()
    image_id = _upload_test_image(client, dataset["id"])

    resp = client.post(f"/api/v1/models/{model['id']}/predict?image_id={image_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["raw_count"] == 3
    assert body["filtered_count"] == 2  # the near-duplicate cone is merged away
    class_names = sorted(p["class_name"] for p in body["predictions"])
    assert class_names == ["ball", "cone"]
    cone = next(p for p in body["predictions"] if p["class_name"] == "cone")
    assert cone["confidence"] == 0.72  # kept the higher-confidence duplicate
    assert all(p["source"] == "auto" for p in body["predictions"])


def test_register_yolo_world_model_via_api(client: TestClient, tmp_path: Path) -> None:
    weights = tmp_path / "yolo_world.pt"
    resp = client.post(
        "/api/v1/models",
        json={"name": "yw_v1", "weights_path": str(weights), "kind": "DETECTOR", "framework": "yolo-world"},
    )
    assert resp.status_code == 201, resp.text
    model = resp.json()
    assert model["is_promptable"] is True
    assert model["framework"] == "yolo-world"


def test_predict_yolo_world_uses_project_class_names(client: TestClient, tmp_path: Path, unique_name: str) -> None:
    project = client.post("/api/v1/projects", json={"name": unique_name}).json()
    client.patch(
        f"/api/v1/projects/{project['id']}",
        json={"class_config": [{"id": 0, "name": "forklift"}]},
    )
    dataset = client.post(f"/api/v1/projects/{project['id']}/datasets", json={"name": "d"}).json()
    image_id = _upload_test_image(client, dataset["id"])

    weights = tmp_path / "yolo_world.pt"
    model = client.post(
        "/api/v1/models",
        json={"name": "yw_v1", "weights_path": str(weights), "kind": "DETECTOR", "framework": "yolo-world"},
    ).json()

    resp = client.post(f"/api/v1/models/{model['id']}/predict?image_id={image_id}")
    assert resp.status_code == 200, resp.text
    predictions = resp.json()["predictions"]
    assert len(predictions) == 1
    assert predictions[0]["class_name"] == "forklift"  # project's class, not the checkpoint's default "person"


def test_predict_unknown_model_404(client: TestClient, unique_name: str) -> None:
    project = client.post("/api/v1/projects", json={"name": unique_name}).json()
    dataset = client.post(f"/api/v1/projects/{project['id']}/datasets", json={"name": "d"}).json()
    image_id = _upload_test_image(client, dataset["id"])

    resp = client.post(
        f"/api/v1/models/00000000-0000-0000-0000-000000000000/predict?image_id={image_id}"
    )
    assert resp.status_code == 400


def test_predict_unknown_image_404(client: TestClient, tmp_path: Path) -> None:
    weights = tmp_path / "detect_v1.pt"
    model = client.post(
        "/api/v1/models",
        json={"name": "detect_v1", "weights_path": str(weights), "kind": "DETECTOR"},
    ).json()
    resp = client.post(
        f"/api/v1/models/{model['id']}/predict?image_id=00000000-0000-0000-0000-000000000000"
    )
    assert resp.status_code == 404
