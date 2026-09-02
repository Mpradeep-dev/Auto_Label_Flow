"""SAM-assisted segmentation: the mask->polygon tracing, the interactive
endpoint, the on-disk checkpoint status, and the download task. The
`ultralytics.SAM` class and the download's HTTP client are both
monkeypatched (house convention: fake the external boundary, keep the
suite offline)."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from app.services.inference import sam_adapter
from app.services.inference.sam_adapter import SamSegmenter, _mask_to_polygon
from app.services.system import sam_models


def _jpeg_bytes(w: int = 64, h: int = 48) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (w, h), color=(70, 90, 110)).save(buf, format="JPEG")
    return buf.getvalue()


# --- _mask_to_polygon ---


def test_mask_to_polygon_normalizes_a_filled_square():
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[10:30, 10:40] = 1  # rows(y) 10..29, cols(x) 10..39

    points = _mask_to_polygon(mask, width=64, height=48)

    assert points is not None
    assert len(points) >= 4
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    assert min(xs) == pytest.approx(10 / 64, abs=1 / 64)
    assert max(xs) == pytest.approx(39 / 64, abs=1 / 64)
    assert min(ys) == pytest.approx(10 / 48, abs=1 / 48)
    assert max(ys) == pytest.approx(29 / 48, abs=1 / 48)


def test_mask_to_polygon_empty_mask_returns_none():
    assert _mask_to_polygon(np.zeros((48, 64), dtype=np.uint8), width=64, height=48) is None


# --- SamSegmenter (ultralytics.SAM faked) ---


class _FakeMasks:
    def __init__(self, xy):
        self.xy = xy
        self.data = [np.ones((1, 1))]  # only its length is inspected when xy is present


class _FakeResult:
    def __init__(self, masks):
        self.masks = masks


class _FakeSAM:
    last_call: dict | None = None

    def __init__(self, weights_path: str) -> None:
        self.weights_path = weights_path

    def predict(self, source, points, labels, device, verbose=False):
        _FakeSAM.last_call = {"points": points, "labels": labels, "device": device}
        return [_FakeResult(_FakeMasks(xy=[[[8, 8], [40, 8], [40, 32], [8, 32]]]))]


class _FakeSAMNoMask:
    def __init__(self, weights_path: str) -> None:
        pass

    def predict(self, source, points, labels, device, verbose=False):
        return [_FakeResult(masks=None)]


def test_segmenter_traces_masks_xy_into_normalized_polygon(tmp_path, monkeypatch):
    import ultralytics

    monkeypatch.setattr(ultralytics, "SAM", _FakeSAM)
    weights = tmp_path / "mobile_sam.pt"
    weights.write_bytes(b"fake")

    segmenter = SamSegmenter(str(weights))
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    points = segmenter.segment(image, points=[(0.5, 0.5)], labels=[1])

    assert points == [[8 / 64, 8 / 48], [40 / 64, 8 / 48], [40 / 64, 32 / 48], [8 / 64, 32 / 48]]
    assert _FakeSAM.last_call["points"] == [[[32, 24]]]  # (0.5,0.5) on a 64x48 image
    assert _FakeSAM.last_call["labels"] == [[1]]


def test_segmenter_returns_none_when_no_mask(tmp_path, monkeypatch):
    import ultralytics

    monkeypatch.setattr(ultralytics, "SAM", _FakeSAMNoMask)
    weights = tmp_path / "mobile_sam.pt"
    weights.write_bytes(b"fake")

    segmenter = SamSegmenter(str(weights))
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    assert segmenter.segment(image, points=[(0.1, 0.1)], labels=[1]) is None


def test_segmenter_rejects_mismatched_points_and_labels(tmp_path, monkeypatch):
    import ultralytics

    monkeypatch.setattr(ultralytics, "SAM", _FakeSAM)
    weights = tmp_path / "mobile_sam.pt"
    weights.write_bytes(b"fake")
    segmenter = SamSegmenter(str(weights))

    with pytest.raises(sam_adapter.SamSegmentationError):
        segmenter.segment(np.zeros((48, 64, 3), dtype=np.uint8), points=[(0.1, 0.1)], labels=[1, 0])


# --- sam_models status/remove ---


def test_sam_model_status_reflects_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(sam_models.settings, "ARTIFACTS_DIR", tmp_path)

    before = sam_models.status("sam-lite")
    assert before.installed is False
    assert before.size_bytes is None

    sam_models.weights_path("sam-lite").write_bytes(b"x" * 100)
    after = sam_models.status("sam-lite")
    assert after.installed is True
    assert after.size_bytes == 100
    assert sam_models.any_installed() is True

    sam_models.remove("sam-lite")
    assert sam_models.is_installed("sam-lite") is False
    assert sam_models.any_installed() is False


# --- /images/{id}/segment endpoint ---


def test_segment_endpoint_rejects_unknown_variant(client: TestClient, unique_name: str) -> None:
    project = client.post("/api/v1/projects", json={"name": unique_name}).json()
    dataset = client.post(f"/api/v1/projects/{project['id']}/datasets", json={"name": "d"}).json()
    image = client.post(
        f"/api/v1/datasets/{dataset['id']}/images", files={"file": ("f.jpg", _jpeg_bytes(), "image/jpeg")}
    ).json()

    resp = client.post(f"/api/v1/images/{image['id']}/segment", json={"variant": "nope", "points": [[0.5, 0.5]]})
    assert resp.status_code == 400
    assert "unknown" in resp.json()["detail"].lower()


def test_segment_endpoint_rejects_when_not_downloaded(client: TestClient, unique_name: str, monkeypatch) -> None:
    monkeypatch.setattr(sam_models, "is_installed", lambda name: False)
    project = client.post("/api/v1/projects", json={"name": unique_name}).json()
    dataset = client.post(f"/api/v1/projects/{project['id']}/datasets", json={"name": "d"}).json()
    image = client.post(
        f"/api/v1/datasets/{dataset['id']}/images", files={"file": ("f.jpg", _jpeg_bytes(), "image/jpeg")}
    ).json()

    resp = client.post(
        f"/api/v1/images/{image['id']}/segment", json={"variant": "sam-lite", "points": [[0.5, 0.5]]}
    )
    assert resp.status_code == 400
    assert "download" in resp.json()["detail"].lower()


def test_segment_endpoint_returns_polygon_when_installed(client: TestClient, unique_name: str, monkeypatch) -> None:
    import app.api.v1.images as images_mod

    monkeypatch.setattr(sam_models, "is_installed", lambda name: True)

    class _FakeSegmenter:
        def segment(self, image, points, labels):
            return [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]]

    monkeypatch.setattr(images_mod, "get_segmenter", lambda weights_path: _FakeSegmenter())

    project = client.post("/api/v1/projects", json={"name": unique_name}).json()
    dataset = client.post(f"/api/v1/projects/{project['id']}/datasets", json={"name": "d"}).json()
    image = client.post(
        f"/api/v1/datasets/{dataset['id']}/images", files={"file": ("f.jpg", _jpeg_bytes(), "image/jpeg")}
    ).json()

    resp = client.post(
        f"/api/v1/images/{image['id']}/segment", json={"variant": "sam-lite", "points": [[0.5, 0.5]]}
    )
    assert resp.status_code == 200
    assert resp.json()["points"] == [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]]

    # The resulting points feed straight into the normal polygon-create
    # path, exactly like a hand-drawn polygon.
    created = client.post(
        "/api/v1/annotations",
        json={
            "image_id": image["id"],
            "class_id": 0,
            "class_name": "cone",
            "shape_type": "POLYGON",
            "points": resp.json()["points"],
        },
    )
    assert created.status_code == 201, created.text
    assert created.json()["source"] == "HUMAN"


def test_segment_endpoint_404_for_missing_image(client: TestClient) -> None:
    import uuid

    resp = client.post(f"/api/v1/images/{uuid.uuid4()}/segment", json={"variant": "sam-lite", "points": [[0.5, 0.5]]})
    assert resp.status_code == 404


def test_system_sam_models_listing(client: TestClient) -> None:
    resp = client.get("/api/v1/system/sam-models")
    assert resp.status_code == 200
    names = {m["name"] for m in resp.json()}
    assert names == {"sam-lite", "sam-full"}
    assert all(m["installed"] is False for m in resp.json())


# --- download task ---


class _FakeStreamResponse:
    def __init__(self, content: bytes, status_code: int = 200, headers: dict | None = None, is_redirect: bool = False):
        self._content = content
        self.status_code = status_code
        self.headers = headers or {}
        self.is_redirect = is_redirect

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_bytes(self, chunk_size: int = 1024 * 1024):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]


class _FakeHttpxClient:
    def __init__(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def stream(self, method, url):
        return _FakeStreamResponse(b"fake-checkpoint-bytes", headers={"content-length": "21"})


def test_download_sam_model_writes_file_and_reports_progress(tmp_path, monkeypatch):
    import app.workers.tasks.sam_download as sam_download_mod

    monkeypatch.setattr(sam_models.settings, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(sam_download_mod, "assert_public_host", lambda url: None)
    monkeypatch.setattr(sam_download_mod.httpx, "Client", _FakeHttpxClient)

    sam_download_mod.download_sam_model("sam-lite")

    assert sam_models.weights_path("sam-lite").read_bytes() == b"fake-checkpoint-bytes"
    progress = sam_download_mod.get_download_progress("sam-lite")
    assert progress["state"] == "done"
    assert progress["downloaded"] == len(b"fake-checkpoint-bytes")


def test_download_sam_model_unknown_variant_fails_cleanly(monkeypatch):
    import app.workers.tasks.sam_download as sam_download_mod

    sam_download_mod.download_sam_model("nope")
    progress = sam_download_mod.get_download_progress("nope")
    assert progress["state"] == "failed"
