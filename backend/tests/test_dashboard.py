"""Dataset statistics / error analysis / model metrics — the acceptance-
rate math is the one worth testing precisely: accepted + corrected +
rejected must exactly partition every AUTO-originated annotation (PLAN
spec section 13's worked example: 9,800 + 1,850 + 800 = 12,450)."""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage


def _jpeg() -> bytes:
    img = PILImage.new("RGB", (64, 48), color=(70, 70, 70))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _upload_image(client: TestClient, dataset_id: str) -> dict:
    return client.post(
        f"/api/v1/datasets/{dataset_id}/images", files={"file": ("f.jpg", _jpeg(), "image/jpeg")}
    ).json()


def test_acceptance_rate_partitions_auto_predictions(client: TestClient, unique_name: str) -> None:
    project = client.post("/api/v1/projects", json={"name": unique_name}).json()
    dataset = client.post(f"/api/v1/projects/{project['id']}/datasets", json={"name": "d"}).json()

    # image 1: AUTO box left untouched -> "accepted"
    img1 = _upload_image(client, dataset["id"])
    ann1 = client.post(
        "/api/v1/annotations",
        json={"image_id": img1["id"], "class_id": 1, "class_name": "cone", "x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2},
    ).json()
    # created as HUMAN by that endpoint — bypass to simulate an AUTO origin directly via the DB isn't
    # available here, so use the auto-annotate path instead for a real AUTO row:
    assert ann1["source"] == "HUMAN"  # sanity: this one does NOT count toward auto_label_acceptance

    resp = client.get(f"/api/v1/datasets/{dataset['id']}/statistics")
    assert resp.status_code == 200
    stats = resp.json()
    assert stats["total_images"] == 1
    assert stats["annotations_by_source"]["HUMAN"] == 1
    # No AUTO predictions were made in this dataset — acceptance rate is undefined, not zero.
    assert stats["auto_label_acceptance"]["total_auto_predictions"] == 0
    assert stats["auto_label_acceptance"]["acceptance_rate"] is None


def test_acceptance_rate_with_real_auto_predictions(client: TestClient, unique_name: str, tmp_path, monkeypatch) -> None:
    from app.services.inference import registry
    from app.services.inference.detector import Detection

    class _Fake:
        def __init__(self, weights_path):
            pass

        @property
        def class_names(self):
            return {0: "ball", 1: "cone"}

        def predict(self, image, conf=0.2, iou=0.7, imgsz=640):
            return [
                Detection(class_id=1, class_name="cone", confidence=0.7, x1=0.1, y1=0.1, x2=0.2, y2=0.2),
                Detection(class_id=1, class_name="cone", confidence=0.3, x1=0.5, y1=0.5, x2=0.6, y2=0.6),
                Detection(class_id=1, class_name="cone", confidence=0.3, x1=0.7, y1=0.7, x2=0.8, y2=0.8),
            ]

    registry.clear_cache()
    monkeypatch.setattr(registry, "UltralyticsDetectionModel", _Fake)

    project = client.post("/api/v1/projects", json={"name": unique_name}).json()
    dataset = client.post(f"/api/v1/projects/{project['id']}/datasets", json={"name": "d"}).json()
    weights = tmp_path / "m.pt"
    model = client.post("/api/v1/models", json={"name": "m", "weights_path": str(weights), "kind": "DETECTOR"}).json()

    img = _upload_image(client, dataset["id"])
    anns = client.post(f"/api/v1/images/{img['id']}/auto-annotate", json={"model_id": model["id"]}).json()
    assert len(anns) == 3

    # Leave anns[0] untouched (accepted), correct anns[1] (corrected), delete anns[2] (rejected).
    client.put(f"/api/v1/annotations/{anns[1]['id']}", json={"x1": 0.55})
    client.request("DELETE", f"/api/v1/annotations/{anns[2]['id']}", json={"error_category": "FALSE_POSITIVE", "error_reason": "PLAYER_FOOT"})

    stats = client.get(f"/api/v1/datasets/{dataset['id']}/statistics").json()
    acceptance = stats["auto_label_acceptance"]
    assert acceptance["total_auto_predictions"] == 3
    assert acceptance["accepted"] == 1
    assert acceptance["corrected"] == 1
    assert acceptance["rejected"] == 1
    assert acceptance["acceptance_rate"] == pytest.approx(100 / 3, abs=0.1)

    registry.clear_cache()


def test_error_analysis_aggregates_deletion_reasons(client: TestClient, unique_name: str) -> None:
    project = client.post("/api/v1/projects", json={"name": unique_name}).json()
    dataset = client.post(f"/api/v1/projects/{project['id']}/datasets", json={"name": "d"}).json()
    img = _upload_image(client, dataset["id"])
    ann = client.post(
        "/api/v1/annotations",
        json={"image_id": img["id"], "class_id": 1, "class_name": "cone", "x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2},
    ).json()
    client.request(
        "DELETE",
        f"/api/v1/annotations/{ann['id']}",
        json={"error_category": "FALSE_POSITIVE", "error_reason": "PLAYER_FOOT"},
    )

    resp = client.get(f"/api/v1/datasets/{dataset['id']}/error-analysis")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_categorized_deletions"] == 1
    assert body["by_category"]["FALSE_POSITIVE"] == 1
    assert body["by_reason"]["PLAYER_FOOT"] == 1


def test_dataset_statistics_missing_dataset_404(client: TestClient) -> None:
    resp = client.get("/api/v1/datasets/00000000-0000-0000-0000-000000000000/statistics")
    assert resp.status_code == 404


def test_update_model_metrics(client: TestClient, tmp_path, unique_name: str, monkeypatch) -> None:
    weights = tmp_path / "m.pt"
    from app.services.inference import registry

    class _Fake:
        def __init__(self, weights_path):
            pass

        @property
        def class_names(self):
            return {0: "ball"}

        def predict(self, *a, **k):
            return []

    registry.clear_cache()
    monkeypatch.setattr(registry, "UltralyticsDetectionModel", _Fake)

    model = client.post(
        "/api/v1/models", json={"name": unique_name, "weights_path": str(weights), "kind": "DETECTOR"}
    ).json()
    resp = client.put(
        f"/api/v1/models/{model['id']}/metrics",
        json={"metrics": {"map50": 0.86, "cone_precision": 0.91, "cone_recall": 0.83}},
    )
    assert resp.status_code == 200
    assert resp.json()["metrics"]["map50"] == 0.86
    registry.clear_cache()
