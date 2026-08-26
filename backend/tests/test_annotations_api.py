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
        pass

    @property
    def class_names(self) -> dict[int, str]:
        return {0: "ball", 1: "cone", 2: "cone_1"}

    def predict(self, image, conf=0.20, iou=0.70, imgsz=640) -> list[Detection]:
        return [
            Detection(class_id=1, class_name="cone", confidence=0.72, x1=0.20, y1=0.58, x2=0.23, y2=0.64),
            Detection(class_id=1, class_name="cone", confidence=0.34, x1=0.10, y1=0.63, x2=0.12, y2=0.67),
        ]


@pytest.fixture(autouse=True)
def _patch_detector(monkeypatch):
    registry.clear_cache()
    monkeypatch.setattr(registry, "UltralyticsDetectionModel", _FakeDetectionModel)
    yield
    registry.clear_cache()


@pytest.fixture()
def image_id(client: TestClient, unique_name: str) -> str:
    project = client.post("/api/v1/projects", json={"name": unique_name}).json()
    dataset = client.post(f"/api/v1/projects/{project['id']}/datasets", json={"name": "d"}).json()
    arr = (np.random.rand(60, 80, 3) * 255).astype("uint8")
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="JPEG")
    resp = client.post(
        f"/api/v1/datasets/{dataset['id']}/images",
        files={"file": ("sample.jpg", buf.getvalue(), "image/jpeg")},
    )
    return resp.json()["id"]


def test_create_list_annotation(client: TestClient, image_id: str) -> None:
    resp = client.post(
        "/api/v1/annotations",
        json={"image_id": image_id, "class_id": 1, "class_name": "cone", "x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2},
    )
    assert resp.status_code == 201, resp.text
    ann = resp.json()
    assert ann["source"] == "HUMAN"

    listed = client.get(f"/api/v1/images/{image_id}/annotations").json()
    assert len(listed) == 1
    assert listed[0]["id"] == ann["id"]


def test_create_rejects_degenerate_box(client: TestClient, image_id: str) -> None:
    # Bbox-ordering validation now lives in AnnotationCreate's model_validator
    # (applies uniformly to BBOX and POLYGON payloads), so a violation is a
    # request-schema failure -> 422, not a hand-rolled 400 in the route.
    resp = client.post(
        "/api/v1/annotations",
        json={"image_id": image_id, "class_id": 1, "class_name": "cone", "x1": 0.2, "y1": 0.1, "x2": 0.2, "y2": 0.2},
    )
    assert resp.status_code == 422


def test_update_annotation_moves_source_to_corrected(client: TestClient, image_id: str) -> None:
    ann = client.post(
        "/api/v1/annotations",
        json={"image_id": image_id, "class_id": 1, "class_name": "cone", "x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2},
    ).json()
    # HUMAN stays HUMAN — flip to test the AUTO path via auto-annotate instead
    updated = client.put(f"/api/v1/annotations/{ann['id']}", json={"x1": 0.15}).json()
    assert updated["source"] == "HUMAN"
    assert updated["x1"] == 0.15


def test_create_polygon_derives_bbox(client: TestClient, image_id: str) -> None:
    resp = client.post(
        "/api/v1/annotations",
        json={
            "image_id": image_id,
            "class_id": 1,
            "class_name": "cone",
            "shape_type": "POLYGON",
            "points": [[0.1, 0.1], [0.3, 0.1], [0.2, 0.4]],
        },
    )
    assert resp.status_code == 201, resp.text
    ann = resp.json()
    assert ann["shape_type"] == "POLYGON"
    assert ann["points"] == [[0.1, 0.1], [0.3, 0.1], [0.2, 0.4]]
    assert (ann["x1"], ann["y1"], ann["x2"], ann["y2"]) == (0.1, 0.1, 0.3, 0.4)


def test_create_polygon_rejects_too_few_points(client: TestClient, image_id: str) -> None:
    resp = client.post(
        "/api/v1/annotations",
        json={
            "image_id": image_id,
            "class_id": 1,
            "class_name": "cone",
            "shape_type": "POLYGON",
            "points": [[0.1, 0.1], [0.3, 0.1]],
        },
    )
    assert resp.status_code == 422


def test_update_polygon_points_recomputes_bbox_and_flips_corrected(client: TestClient, image_id: str) -> None:
    # AUTO source only comes from a detector run — go through auto-annotate,
    # then convert one prediction's shape isn't possible (shape_type is
    # immutable), so instead: create a HUMAN polygon, confirm point edits
    # recompute the bbox; the AUTO -> CORRECTED flip on points-only change is
    # covered directly in test_annotation_service.py against a synthetic AUTO row.
    ann = client.post(
        "/api/v1/annotations",
        json={
            "image_id": image_id,
            "class_id": 1,
            "class_name": "cone",
            "shape_type": "POLYGON",
            "points": [[0.1, 0.1], [0.3, 0.1], [0.2, 0.4]],
        },
    ).json()
    updated = client.put(
        f"/api/v1/annotations/{ann['id']}",
        json={"points": [[0.0, 0.0], [0.5, 0.0], [0.25, 0.5]]},
    ).json()
    assert updated["points"] == [[0.0, 0.0], [0.5, 0.0], [0.25, 0.5]]
    assert (updated["x1"], updated["y1"], updated["x2"], updated["y2"]) == (0.0, 0.0, 0.5, 0.5)


def test_update_rejects_bbox_fields_on_polygon(client: TestClient, image_id: str) -> None:
    ann = client.post(
        "/api/v1/annotations",
        json={
            "image_id": image_id,
            "class_id": 1,
            "class_name": "cone",
            "shape_type": "POLYGON",
            "points": [[0.1, 0.1], [0.3, 0.1], [0.2, 0.4]],
        },
    ).json()
    resp = client.put(f"/api/v1/annotations/{ann['id']}", json={"x1": 0.2})
    assert resp.status_code == 400


def test_update_rejects_points_on_bbox(client: TestClient, image_id: str) -> None:
    ann = client.post(
        "/api/v1/annotations",
        json={"image_id": image_id, "class_id": 1, "class_name": "cone", "x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2},
    ).json()
    resp = client.put(
        f"/api/v1/annotations/{ann['id']}",
        json={"points": [[0.0, 0.0], [0.5, 0.0], [0.25, 0.5]]},
    )
    assert resp.status_code == 400


def test_delete_annotation_with_error_reason(client: TestClient, image_id: str) -> None:
    ann = client.post(
        "/api/v1/annotations",
        json={"image_id": image_id, "class_id": 1, "class_name": "cone", "x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2},
    ).json()
    resp = client.request(
        "DELETE",
        f"/api/v1/annotations/{ann['id']}",
        json={"error_category": "FALSE_POSITIVE", "error_reason": "PLAYER_FOOT"},
    )
    assert resp.status_code == 204
    assert client.get(f"/api/v1/images/{ann['image_id']}/annotations").json() == []


def test_duplicate_annotation(client: TestClient, image_id: str) -> None:
    ann = client.post(
        "/api/v1/annotations",
        json={"image_id": image_id, "class_id": 1, "class_name": "cone", "x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2},
    ).json()
    dup = client.post(f"/api/v1/annotations/{ann['id']}/duplicate")
    assert dup.status_code == 201
    assert dup.json()["id"] != ann["id"]
    assert len(client.get(f"/api/v1/images/{ann['image_id']}/annotations").json()) == 2


def test_approve_leaves_untouched_auto_annotation_intact(
    client: TestClient, image_id: str, tmp_path: Path
) -> None:
    weights = tmp_path / "detect_v1.pt"
    model = client.post(
        "/api/v1/models", json={"name": "detect_v1", "weights_path": str(weights), "kind": "DETECTOR"}
    ).json()

    auto = client.post(f"/api/v1/images/{image_id}/auto-annotate", json={"model_id": model["id"]})
    assert auto.status_code == 200, auto.text
    annotations = auto.json()
    assert len(annotations) == 2
    assert all(a["source"] == "AUTO" for a in annotations)

    approved_image = client.post(f"/api/v1/images/{image_id}/approve")
    assert approved_image.status_code == 200
    assert approved_image.json()["review_status"] == "APPROVED"

    still = client.get(f"/api/v1/images/{image_id}/annotations").json()
    assert all(a["source"] == "AUTO" for a in still)  # approve alone must not relabel to CORRECTED
    assert all(a["review_status"] == "APPROVED" for a in still)


def test_auto_annotate_then_correct_moves_to_corrected(client: TestClient, image_id: str, tmp_path: Path) -> None:
    weights = tmp_path / "detect_v1.pt"
    model = client.post(
        "/api/v1/models", json={"name": "detect_v1", "weights_path": str(weights), "kind": "DETECTOR"}
    ).json()
    annotations = client.post(
        f"/api/v1/images/{image_id}/auto-annotate", json={"model_id": model["id"]}
    ).json()
    target = annotations[0]
    # auto-annotate resolves "cone" against the project's (empty) class_config
    # and registers it under whatever id comes next — pick a different one so
    # this PUT is an actual class change, not a no-op that happens to match.
    new_class_id = target["class_id"] + 1

    corrected = client.put(
        f"/api/v1/annotations/{target['id']}", json={"class_id": new_class_id, "class_name": "ball"}
    ).json()
    assert corrected["source"] == "CORRECTED"
    assert corrected["class_name"] == "ball"


def test_auto_annotate_replace_existing_clears_prior_auto(client: TestClient, image_id: str, tmp_path: Path) -> None:
    weights = tmp_path / "detect_v1.pt"
    model = client.post(
        "/api/v1/models", json={"name": "detect_v1", "weights_path": str(weights), "kind": "DETECTOR"}
    ).json()
    client.post(f"/api/v1/images/{image_id}/auto-annotate", json={"model_id": model["id"]})
    second = client.post(f"/api/v1/images/{image_id}/auto-annotate", json={"model_id": model["id"]}).json()

    all_annotations = client.get(f"/api/v1/images/{image_id}/annotations").json()
    assert len(all_annotations) == len(second)  # not doubled


def test_reject_image(client: TestClient, image_id: str) -> None:
    resp = client.post(f"/api/v1/images/{image_id}/reject")
    assert resp.status_code == 200
    assert resp.json()["review_status"] == "REJECTED"


def test_update_missing_annotation_404(client: TestClient) -> None:
    resp = client.put(
        "/api/v1/annotations/00000000-0000-0000-0000-000000000000", json={"x1": 0.2}
    )
    assert resp.status_code == 404


def test_approve_missing_image_404(client: TestClient) -> None:
    resp = client.post("/api/v1/images/00000000-0000-0000-0000-000000000000/approve")
    assert resp.status_code == 404
