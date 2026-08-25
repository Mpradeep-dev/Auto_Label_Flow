"""API-level tests for the quality analysis + review queue endpoints, using
fake detector/pose models (same pattern as test_models_api.py) so no real
weights are loaded — house convention."""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from app.services.inference import registry
from app.services.inference.detector import Detection, Keypoint, PoseResult


class _FakeDetectionModel:
    def __init__(self, weights_path: str) -> None:
        pass

    @property
    def class_names(self) -> dict[int, str]:
        return {0: "ball", 1: "cone", 2: "cone_1"}

    def predict(self, image, conf=0.20, iou=0.70, imgsz=640) -> list[Detection]:
        # A cone planted right where the fake pose model will report an ankle.
        return [Detection(class_id=1, class_name="cone", confidence=0.34, x1=0.095, y1=0.62, x2=0.125, y2=0.66)]


class _FakePoseModel:
    def __init__(self, weights_path: str) -> None:
        pass

    def predict(self, image, conf=0.50, imgsz=640) -> list[PoseResult]:
        kpts = [Keypoint(x=0, y=0, confidence=0.0) for _ in range(17)]
        kpts[15] = Keypoint(x=0.11, y=0.64, confidence=0.9)  # left ankle, right at the fake cone
        kpts[16] = Keypoint(x=0.9, y=0.64, confidence=0.9)
        kpts[11] = Keypoint(x=0.11, y=0.40, confidence=0.9)  # left hip, for leg-length body_scale
        kpts[12] = Keypoint(x=0.9, y=0.40, confidence=0.9)
        return [PoseResult(x1=0.05, y1=0.35, x2=0.30, y2=0.68, confidence=0.9, keypoints=kpts)]


class _FakeRawYOLO:
    """Stands in for `ultralytics.YOLO` itself — used only by
    `registry._probe_class_names`'s POSE branch, which (unlike the
    DETECTOR branch) calls the raw ultralytics constructor directly to
    read `.names` at registration time."""

    def __init__(self, weights_path: str) -> None:
        self.names = {0: "person"}


@pytest.fixture(autouse=True)
def _patch_models(monkeypatch):
    registry.clear_cache()
    monkeypatch.setattr(registry, "UltralyticsDetectionModel", _FakeDetectionModel)
    monkeypatch.setattr(registry, "UltralyticsPoseModel", _FakePoseModel)
    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", _FakeRawYOLO)
    yield
    registry.clear_cache()


@pytest.fixture()
def project_with_pose_and_image(client: TestClient, unique_name: str, tmp_path) -> tuple[dict, dict]:
    project = client.post("/api/v1/projects", json={"name": unique_name}).json()

    detect_weights = tmp_path / "detect.pt"
    detect_model = client.post(
        "/api/v1/models", json={"name": "detect", "weights_path": str(detect_weights), "kind": "DETECTOR"}
    ).json()
    pose_weights = tmp_path / "pose.pt"
    pose_model = client.post(
        "/api/v1/models", json={"name": "pose", "weights_path": str(pose_weights), "kind": "POSE"}
    ).json()

    client.patch(
        f"/api/v1/projects/{project['id']}",
        json={
            "class_config": [{"id": 0, "name": "ball"}, {"id": 1, "name": "cone"}, {"id": 2, "name": "cone_1"}],
        },
    )
    # pose_model_id isn't in ProjectUpdate's typed fields by name coincidence — it is; set directly.
    client.patch(f"/api/v1/projects/{project['id']}", json={"pose_model_id": pose_model["id"]})

    dataset = client.post(f"/api/v1/projects/{project['id']}/datasets", json={"name": "d"}).json()
    img = PILImage.new("RGB", (100, 100), color=(90, 90, 90))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    image = client.post(
        f"/api/v1/datasets/{dataset['id']}/images", files={"file": ("f.jpg", buf.getvalue(), "image/jpeg")}
    ).json()

    client.post(f"/api/v1/images/{image['id']}/auto-annotate", json={"model_id": detect_model["id"], "conf": 0.2})

    return project, image


def test_analyze_quality_flags_the_foot_false_positive(client: TestClient, project_with_pose_and_image) -> None:
    _, image = project_with_pose_and_image
    resp = client.post(f"/api/v1/images/{image['id']}/analyze-quality")
    assert resp.status_code == 200, resp.text
    flags = resp.json()
    flag_types = {f["flag_type"] for f in flags}
    assert "SUSPICIOUS_CONE" in flag_types
    assert "CONE_NEAR_PLAYER" in flag_types
    assert "LOW_CONFIDENCE" in flag_types  # confidence 0.34 < the 0.65 cone threshold

    suspicious = next(f for f in flags if f["flag_type"] == "SUSPICIOUS_CONE")
    assert suspicious["details"]["distance_bl"] < 0.75


def test_analyze_quality_updates_difficulty_score(client: TestClient, project_with_pose_and_image) -> None:
    _, image = project_with_pose_and_image
    client.post(f"/api/v1/images/{image['id']}/analyze-quality")
    updated = client.get(f"/api/v1/images/{image['id']}").json()
    assert updated["difficulty_score"] is not None
    assert updated["difficulty_score"] > 0


def test_list_image_flags(client: TestClient, project_with_pose_and_image) -> None:
    _, image = project_with_pose_and_image
    client.post(f"/api/v1/images/{image['id']}/analyze-quality")
    flags = client.get(f"/api/v1/images/{image['id']}/flags").json()
    assert len(flags) >= 1


def test_resolve_flag(client: TestClient, project_with_pose_and_image) -> None:
    _, image = project_with_pose_and_image
    flags = client.post(f"/api/v1/images/{image['id']}/analyze-quality").json()
    flag_id = flags[0]["id"]

    resp = client.post(f"/api/v1/annotation-flags/{flag_id}/resolve", json={"resolution": "CONFIRMED_FP"})
    assert resp.status_code == 200
    assert resp.json()["resolution"] == "CONFIRMED_FP"


def test_resolve_missing_flag_404(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/annotation-flags/00000000-0000-0000-0000-000000000000/resolve",
        json={"resolution": "CONFIRMED_OK"},
    )
    assert resp.status_code == 404


def test_review_queue_orders_by_difficulty_descending(client: TestClient, project_with_pose_and_image) -> None:
    project, image = project_with_pose_and_image
    client.post(f"/api/v1/images/{image['id']}/analyze-quality")

    # A second, clean image with no flags — should rank below the suspicious one.
    dataset_id = image["dataset_id"]
    clean_img = PILImage.new("RGB", (100, 100), color=(10, 10, 10))
    buf = io.BytesIO()
    clean_img.save(buf, format="JPEG")
    clean = client.post(
        f"/api/v1/datasets/{dataset_id}/images", files={"file": ("clean.jpg", buf.getvalue(), "image/jpeg")}
    ).json()

    resp = client.get(f"/api/v1/review/queue?project_id={project['id']}")
    assert resp.status_code == 200
    body = resp.json()
    image_ids_in_order = [item["image_id"] for item in body["items"]]
    assert image_ids_in_order.index(image["id"]) < image_ids_in_order.index(clean["id"])


def test_review_queue_filters_by_flag_type(client: TestClient, project_with_pose_and_image) -> None:
    project, image = project_with_pose_and_image
    client.post(f"/api/v1/images/{image['id']}/analyze-quality")

    resp = client.get(f"/api/v1/review/queue?project_id={project['id']}&flag_type=SUSPICIOUS_CONE")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert any(item["image_id"] == image["id"] for item in items)

    resp2 = client.get(f"/api/v1/review/queue?project_id={project['id']}&flag_type=TEMPORAL_ANOMALY")
    items2 = resp2.json()["items"]
    assert not any(item["image_id"] == image["id"] for item in items2)


def test_review_queue_splits_pending_from_approved(client: TestClient, project_with_pose_and_image) -> None:
    """The review workflow is split into two buckets by `review_status` —
    an image approved by a human must disappear from the PENDING bucket
    and show up in APPROVED, never both and never neither."""
    project, image = project_with_pose_and_image

    pending = client.get(f"/api/v1/review/queue?project_id={project['id']}&review_status=PENDING").json()
    assert any(item["image_id"] == image["id"] for item in pending["items"])
    approved = client.get(f"/api/v1/review/queue?project_id={project['id']}&review_status=APPROVED").json()
    assert not any(item["image_id"] == image["id"] for item in approved["items"])

    resp = client.post(f"/api/v1/images/{image['id']}/approve")
    assert resp.status_code == 200

    pending_after = client.get(f"/api/v1/review/queue?project_id={project['id']}&review_status=PENDING").json()
    assert not any(item["image_id"] == image["id"] for item in pending_after["items"])
    approved_after = client.get(f"/api/v1/review/queue?project_id={project['id']}&review_status=APPROVED").json()
    assert any(item["image_id"] == image["id"] for item in approved_after["items"])
    assert next(i for i in approved_after["items"] if i["image_id"] == image["id"])["review_status"] == "APPROVED"
