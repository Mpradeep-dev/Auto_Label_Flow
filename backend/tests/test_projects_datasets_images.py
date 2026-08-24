from __future__ import annotations

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image as PILImage


def _make_jpeg_bytes(width: int = 64, height: int = 48) -> bytes:
    arr = (np.random.rand(height, width, 3) * 255).astype("uint8")
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def test_health(client: TestClient) -> None:
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"


def test_create_and_get_project(client: TestClient, unique_name: str) -> None:
    resp = client.post("/api/v1/projects", json={"name": unique_name, "description": "d"})
    assert resp.status_code == 201
    project = resp.json()
    assert project["name"] == unique_name
    assert project["class_config"] == []  # nothing hardcoded — populated on model registration

    resp = client.get(f"/api/v1/projects/{project['id']}")
    assert resp.status_code == 200
    assert resp.json()["id"] == project["id"]


def test_duplicate_project_name_gets_unique_slug(client: TestClient, unique_name: str) -> None:
    r1 = client.post("/api/v1/projects", json={"name": unique_name})
    r2 = client.post("/api/v1/projects", json={"name": unique_name})
    assert r1.status_code == 201 and r2.status_code == 201
    assert r1.json()["slug"] != r2.json()["slug"]


def test_get_missing_project_404(client: TestClient) -> None:
    resp = client.get("/api/v1/projects/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


def test_update_project_class_config(client: TestClient, unique_name: str) -> None:
    project = client.post("/api/v1/projects", json={"name": unique_name}).json()
    resp = client.patch(
        f"/api/v1/projects/{project['id']}",
        json={"class_config": [{"id": 0, "name": "ball"}, {"id": 1, "name": "cone"}, {"id": 2, "name": "cone_1"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["class_config"] == [
        {"id": 0, "name": "ball"},
        {"id": 1, "name": "cone"},
        {"id": 2, "name": "cone_1"},
    ]


def test_dataset_lifecycle(client: TestClient, unique_name: str) -> None:
    project = client.post("/api/v1/projects", json={"name": unique_name}).json()

    resp = client.post(f"/api/v1/projects/{project['id']}/datasets", json={"name": "batch-1"})
    assert resp.status_code == 201
    dataset = resp.json()
    assert dataset["project_id"] == project["id"]
    assert dataset["status"] == "ACTIVE"

    listed = client.get(f"/api/v1/projects/{project['id']}/datasets").json()
    assert len(listed) == 1

    stats = client.get(f"/api/v1/datasets/{dataset['id']}/stats").json()
    assert stats == {"total_images": 0, "pending_images": 0, "approved_images": 0, "total_videos": 0}


def test_dataset_under_missing_project_404(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/projects/00000000-0000-0000-0000-000000000000/datasets", json={"name": "x"}
    )
    assert resp.status_code == 404


def test_image_upload_list_get_delete(client: TestClient, unique_name: str) -> None:
    project = client.post("/api/v1/projects", json={"name": unique_name}).json()
    dataset = client.post(f"/api/v1/projects/{project['id']}/datasets", json={"name": "batch-1"}).json()

    jpeg_bytes = _make_jpeg_bytes(80, 60)
    resp = client.post(
        f"/api/v1/datasets/{dataset['id']}/images",
        files={"file": ("sample.jpg", jpeg_bytes, "image/jpeg")},
    )
    assert resp.status_code == 201, resp.text
    image = resp.json()
    assert image["width"] == 80 and image["height"] == 60
    assert image["source_type"] == "UPLOAD"
    assert image["review_status"] == "PENDING"
    assert image["url"].startswith("/media/")

    media_resp = client.get(image["url"])
    assert media_resp.status_code == 200
    assert media_resp.headers["content-type"] in ("image/jpeg", "application/octet-stream")

    page = client.get(f"/api/v1/datasets/{dataset['id']}/images").json()
    assert page["total"] == 1
    assert page["items"][0]["id"] == image["id"]

    stats = client.get(f"/api/v1/datasets/{dataset['id']}/stats").json()
    assert stats["total_images"] == 1 and stats["pending_images"] == 1

    got = client.get(f"/api/v1/images/{image['id']}")
    assert got.status_code == 200

    deleted = client.delete(f"/api/v1/images/{image['id']}")
    assert deleted.status_code == 204
    assert client.get(f"/api/v1/images/{image['id']}").status_code == 404


def test_image_upload_rejects_bad_extension(client: TestClient, unique_name: str) -> None:
    project = client.post("/api/v1/projects", json={"name": unique_name}).json()
    dataset = client.post(f"/api/v1/projects/{project['id']}/datasets", json={"name": "batch-1"}).json()

    resp = client.post(
        f"/api/v1/datasets/{dataset['id']}/images",
        files={"file": ("sample.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code == 400


def test_image_upload_rejects_corrupt_image_bytes(client: TestClient, unique_name: str) -> None:
    project = client.post("/api/v1/projects", json={"name": unique_name}).json()
    dataset = client.post(f"/api/v1/projects/{project['id']}/datasets", json={"name": "batch-1"}).json()

    resp = client.post(
        f"/api/v1/datasets/{dataset['id']}/images",
        files={"file": ("sample.jpg", b"totally not jpeg bytes", "image/jpeg")},
    )
    assert resp.status_code == 400
