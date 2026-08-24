from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image as PILImage


def _create_approved_image(client: TestClient, dataset_id: str) -> str:
    img = PILImage.new("RGB", (64, 48), color=(50, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    resp = client.post(
        f"/api/v1/datasets/{dataset_id}/images", files={"file": ("f.jpg", buf.getvalue(), "image/jpeg")}
    )
    image = resp.json()
    client.post(f"/api/v1/images/{image['id']}/approve")
    return image["id"]


def test_create_version_and_export_via_api(client: TestClient, unique_name: str) -> None:
    project = client.post(
        "/api/v1/projects",
        json={"name": unique_name},
    ).json()
    client.patch(
        f"/api/v1/projects/{project['id']}",
        json={"class_config": [{"id": 0, "name": "ball"}, {"id": 1, "name": "cone"}]},
    )
    dataset = client.post(f"/api/v1/projects/{project['id']}/datasets", json={"name": "d"}).json()
    image_id = _create_approved_image(client, dataset["id"])
    client.post(
        "/api/v1/annotations",
        json={"image_id": image_id, "class_id": 1, "class_name": "cone", "x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.3},
    )

    resp = client.post(f"/api/v1/datasets/{dataset['id']}/versions", json={})
    assert resp.status_code == 201, resp.text
    version = resp.json()
    assert version["version_number"] == 1
    assert version["status"] == "DRAFT"
    assert version["total_images"] == 1

    listed = client.get(f"/api/v1/datasets/{dataset['id']}/versions").json()
    assert len(listed) == 1

    exported = client.post(f"/api/v1/versions/{version['id']}/export")
    assert exported.status_code == 200, exported.text
    body = exported.json()
    assert body["status"] == "EXPORTED"
    assert body["download_url"] is not None

    download = client.get(body["download_url"])
    assert download.status_code == 200
    # Exact zip content-type string is OS-dependent (Windows' mimetypes
    # module reports x-zip-compressed); what matters is it's a zip variant.
    assert "zip" in download.headers["content-type"]


def test_create_version_with_no_approved_images_400(client: TestClient, unique_name: str) -> None:
    project = client.post("/api/v1/projects", json={"name": unique_name}).json()
    dataset = client.post(f"/api/v1/projects/{project['id']}/datasets", json={"name": "d"}).json()
    resp = client.post(f"/api/v1/datasets/{dataset['id']}/versions", json={})
    assert resp.status_code == 400


def test_get_missing_version_404(client: TestClient) -> None:
    resp = client.get("/api/v1/versions/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404
