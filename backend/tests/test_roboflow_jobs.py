"""Roboflow import/export as background jobs (progress-bar follow-on to
`test_integrations.py`'s connect/disconnect coverage). Uses `real_client`/
`real_db_session` — same reasoning as `test_training_jobs.py` and
`test_inference_jobs.py`: the Celery task opens its own DB session via
`SessionLocal()`, which never sees `client`'s uncommitted outer
transaction. The Roboflow SDK itself is monkeypatched (house convention:
fake the external boundary, keep the suite offline); Celery runs eager in
test env, so these hit COMPLETED synchronously within the request."""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from app.workers.progress import get_progress


def _jpeg_bytes() -> bytes:
    img = PILImage.new("RGB", (64, 48), color=(80, 80, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class _FakeDownloadResult:
    def __init__(self, location: str) -> None:
        self.location = location


class _FakeRoboflowVersion:
    def __init__(self, version: int) -> None:
        self.version = version

    def download(self, fmt: str, location: str) -> _FakeDownloadResult:
        loc = Path(location)
        images_dir = loc / "train" / "images"
        labels_dir = loc / "train" / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        (images_dir / "img1.jpg").write_bytes(_jpeg_bytes())
        (labels_dir / "img1.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (loc / "data.yaml").write_text("names: ['cone']\n", encoding="utf-8")
        return _FakeDownloadResult(str(loc))


class _FakeRoboflowProject:
    def __init__(self, slug: str) -> None:
        self.slug = slug
        self.uploads: list[tuple[str, str | None, str]] = []

    def version(self, v: int) -> _FakeRoboflowVersion:
        return _FakeRoboflowVersion(v)

    def upload(self, *, image_path: str, annotation_path: str | None, annotation_labelmap: str, split: str) -> None:
        self.uploads.append((image_path, annotation_path, split))

    # Raw (unversioned) pull path — no `.version()` here, only `search()`
    # over the project's raw uploaded images plus `image()` per-item detail.
    # Two items on purpose — one already labeled, one not — so tests can
    # exercise both "pull everything" and "unannotated_only" against the
    # same fake project.
    def search(self, *, offset: int = 0, limit: int = 100, fields: list[str] | None = None) -> list[dict]:
        if offset > 0:
            return []
        return [
            {"id": "raw-img-1", "name": "raw1.jpg", "annotations": {"count": 1, "classes": {"cone": 1}}},
            {"id": "raw-img-2", "name": "raw2.jpg", "annotations": {"count": 0, "classes": {}}},
        ]

    def image(self, image_id: str) -> dict:
        if image_id == "raw-img-1":
            return {
                "id": image_id,
                "name": "raw1.jpg",
                "annotation": {
                    "width": 64,
                    "height": 48,
                    "boxes": [{"label": "cone", "x": "32.0", "y": "24.0", "width": "10.0", "height": "10.0"}],
                },
                "urls": {"original": "http://fake-roboflow-cdn.test/raw1.jpg"},
            }
        return {
            "id": image_id,
            "name": "raw2.jpg",
            "annotation": {"width": 64, "height": 48, "boxes": []},
            "urls": {"original": "http://fake-roboflow-cdn.test/raw2.jpg"},
        }


class _FakeHTTPResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        pass


class _FakeRoboflowWorkspace:
    def project(self, slug: str) -> _FakeRoboflowProject:
        return _FakeRoboflowProject(slug)


class _FakeRoboflow:
    def __init__(self, api_key: str) -> None:
        if api_key != "good-key":
            raise ValueError("Invalid API key")
        self.api_key = api_key
        self.current_workspace = "my-workspace"

    def workspace(self, the_workspace: str | None = None) -> _FakeRoboflowWorkspace:
        return _FakeRoboflowWorkspace()


@pytest.fixture()
def connected_roboflow(real_client: TestClient, monkeypatch):
    import roboflow

    monkeypatch.setattr(roboflow, "Roboflow", _FakeRoboflow)
    resp = real_client.post("/api/v1/integrations/roboflow", json={"api_key": "good-key"})
    assert resp.status_code == 200, resp.text
    return real_client


def test_roboflow_import_job_completes_and_creates_dataset(
    connected_roboflow: TestClient, unique_name: str
) -> None:
    project_id = connected_roboflow.post("/api/v1/projects", json={"name": unique_name}).json()["id"]

    resp = connected_roboflow.post(
        f"/api/v1/projects/{project_id}/import/roboflow",
        json={"workspace": "my-workspace", "project": "cones", "version": 1},
    )
    assert resp.status_code == 202, resp.text
    job = resp.json()
    assert job["kind"] == "IMPORT"
    assert job["status"] == "COMPLETED"
    assert job["total_items"] == 1
    assert job["processed_items"] == 1
    assert job["result_dataset_id"] is not None
    assert job["error"] is None

    fetched = connected_roboflow.get(f"/api/v1/integrations/roboflow/jobs/{job['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "COMPLETED"

    progress = get_progress(job["id"])
    assert progress is not None
    assert progress.status == "COMPLETED"
    assert progress.current == 1

    dataset = connected_roboflow.get(f"/api/v1/datasets/{job['result_dataset_id']}")
    assert dataset.status_code == 200
    stats = connected_roboflow.get(f"/api/v1/datasets/{job['result_dataset_id']}/stats").json()
    assert stats["total_images"] == 1
    assert stats["approved_images"] == 1


def test_roboflow_import_job_raw_pull_when_no_version(
    connected_roboflow: TestClient, monkeypatch, unique_name: str
) -> None:
    """No `version` in the request — the project has nothing generated in
    Roboflow yet, so the job should fall back to `import_roboflow_raw_project`
    (raw `search()`/`image()` pull) instead of `.download()`."""
    import app.services.integrations.roboflow_import as roboflow_import_module

    def _fake_get(url: str, timeout: int = 30) -> _FakeHTTPResponse:
        return _FakeHTTPResponse(_jpeg_bytes())

    monkeypatch.setattr(roboflow_import_module.requests, "get", _fake_get)

    project_id = connected_roboflow.post("/api/v1/projects", json={"name": unique_name}).json()["id"]

    resp = connected_roboflow.post(
        f"/api/v1/projects/{project_id}/import/roboflow",
        json={"workspace": "my-workspace", "project": "ground"},
    )
    assert resp.status_code == 202, resp.text
    job = resp.json()
    assert job["kind"] == "IMPORT"
    assert job["version"] is None
    assert job["unannotated_only"] is False
    assert job["status"] == "COMPLETED"
    assert job["total_items"] == 2
    assert job["processed_items"] == 2
    assert job["result_dataset_id"] is not None
    assert job["error"] is None

    dataset_id = job["result_dataset_id"]
    stats = connected_roboflow.get(f"/api/v1/datasets/{dataset_id}/stats").json()
    assert stats["total_images"] == 2
    # Unlike a versioned pull, a raw pull is not curated ground truth —
    # images land pending review, not auto-approved.
    assert stats["approved_images"] == 0

    project = connected_roboflow.get(f"/api/v1/projects/{project_id}").json()
    assert any(c["name"] == "cone" for c in project["class_config"])


def test_roboflow_import_job_raw_pull_unannotated_only(
    connected_roboflow: TestClient, monkeypatch, unique_name: str
) -> None:
    """`unannotated_only: true` narrows the raw pull to the one fake item
    with zero existing Roboflow annotations (raw-img-2) — raw-img-1, which
    already has a box, must be skipped entirely."""
    import app.services.integrations.roboflow_import as roboflow_import_module

    def _fake_get(url: str, timeout: int = 30) -> _FakeHTTPResponse:
        return _FakeHTTPResponse(_jpeg_bytes())

    monkeypatch.setattr(roboflow_import_module.requests, "get", _fake_get)

    project_id = connected_roboflow.post("/api/v1/projects", json={"name": unique_name}).json()["id"]

    resp = connected_roboflow.post(
        f"/api/v1/projects/{project_id}/import/roboflow",
        json={"workspace": "my-workspace", "project": "ground", "unannotated_only": True},
    )
    assert resp.status_code == 202, resp.text
    job = resp.json()
    assert job["unannotated_only"] is True
    assert job["status"] == "COMPLETED"
    assert job["total_items"] == 1
    assert job["processed_items"] == 1

    dataset_id = job["result_dataset_id"]
    images = connected_roboflow.get(f"/api/v1/datasets/{dataset_id}/images").json()["items"]
    assert len(images) == 1
    assert images[0]["original_filename"] == "raw2.jpg"
    annotations = connected_roboflow.get(f"/api/v1/images/{images[0]['id']}/annotations").json()
    assert annotations == []


def test_roboflow_import_job_cancel_stops_early(
    connected_roboflow: TestClient, real_db_session, monkeypatch, unique_name: str
) -> None:
    """Setting the cancel flag before the task runs (same technique as
    `test_training_jobs.py`'s cancel test) proves the per-item
    `should_cancel()` check in `import_roboflow_raw_project` is real, not
    unreachable plumbing — the job must stop before processing anything
    and land CANCELLED, not COMPLETED."""
    import uuid

    import app.services.integrations.roboflow_import as roboflow_import_module
    from app.models.roboflow_job import RoboflowJob, RoboflowJobKind, RoboflowJobStatus
    from app.workers.progress import request_cancel
    from app.workers.tasks.roboflow import run_roboflow_import

    def _fake_get(url: str, timeout: int = 30) -> _FakeHTTPResponse:
        return _FakeHTTPResponse(_jpeg_bytes())

    monkeypatch.setattr(roboflow_import_module.requests, "get", _fake_get)

    project_id = connected_roboflow.post("/api/v1/projects", json={"name": unique_name}).json()["id"]

    job = RoboflowJob(
        project_id=uuid.UUID(project_id),
        kind=RoboflowJobKind.IMPORT,
        status=RoboflowJobStatus.QUEUED,
        workspace="my-workspace",
        project_slug="ground",
        version=None,
    )
    real_db_session.add(job)
    real_db_session.commit()
    real_db_session.refresh(job)

    request_cancel(str(job.id))
    run_roboflow_import(str(job.id))  # direct call, not .delay() — same eager-equivalent pattern as training's test

    real_db_session.refresh(job)
    assert job.status == RoboflowJobStatus.CANCELLED
    assert job.processed_items == 0
    # The dataset row itself is still created (the cancel check runs before
    # the first *image*, not before dataset creation) — cancelling early
    # shouldn't leave the job pointing at nothing.
    assert job.result_dataset_id is not None


def test_roboflow_export_job_cancel_stops_early(
    connected_roboflow: TestClient, real_db_session, approved_version: tuple[str, str]
) -> None:
    import uuid

    from app.models.roboflow_job import RoboflowJob, RoboflowJobKind, RoboflowJobStatus
    from app.workers.progress import request_cancel
    from app.workers.tasks.roboflow import run_roboflow_export

    project_id, version_id = approved_version

    job = RoboflowJob(
        project_id=uuid.UUID(project_id),
        kind=RoboflowJobKind.EXPORT,
        status=RoboflowJobStatus.QUEUED,
        workspace="my-workspace",
        project_slug="cones",
        dataset_version_id=uuid.UUID(version_id),
    )
    real_db_session.add(job)
    real_db_session.commit()
    real_db_session.refresh(job)

    request_cancel(str(job.id))
    run_roboflow_export(str(job.id))

    real_db_session.refresh(job)
    assert job.status == RoboflowJobStatus.CANCELLED
    assert job.processed_items == 0
    assert job.uploaded_count == 0


def test_roboflow_import_job_missing_job_404(connected_roboflow: TestClient) -> None:
    import uuid

    resp = connected_roboflow.get(f"/api/v1/integrations/roboflow/jobs/{uuid.uuid4()}")
    assert resp.status_code == 404


def test_roboflow_import_job_cancel_endpoint_sets_flag(
    connected_roboflow: TestClient, monkeypatch, unique_name: str
) -> None:
    """The HTTP cancel endpoint itself — separate from the task-level
    behavior above — just needs to accept a real job id and 200. Whether
    the flag is honored is `test_roboflow_import_job_cancel_stops_early`'s
    job; this only proves the route exists and doesn't 404/500."""
    import app.services.integrations.roboflow_import as roboflow_import_module

    monkeypatch.setattr(
        roboflow_import_module.requests, "get", lambda url, timeout=30: _FakeHTTPResponse(_jpeg_bytes())
    )

    project_id = connected_roboflow.post("/api/v1/projects", json={"name": unique_name}).json()["id"]
    job = connected_roboflow.post(
        f"/api/v1/projects/{project_id}/import/roboflow",
        json={"workspace": "my-workspace", "project": "ground"},
    ).json()

    resp = connected_roboflow.post(f"/api/v1/integrations/roboflow/jobs/{job['id']}/cancel")
    assert resp.status_code == 200
    assert resp.json()["id"] == job["id"]


@pytest.fixture()
def approved_version(real_client: TestClient, unique_name: str) -> tuple[str, str]:
    project = real_client.post("/api/v1/projects", json={"name": unique_name}).json()
    real_client.patch(
        f"/api/v1/projects/{project['id']}", json={"class_config": [{"id": 0, "name": "cone"}]}
    )
    dataset = real_client.post(f"/api/v1/projects/{project['id']}/datasets", json={"name": "d"}).json()
    image = real_client.post(
        f"/api/v1/datasets/{dataset['id']}/images", files={"file": ("f.jpg", _jpeg_bytes(), "image/jpeg")}
    ).json()
    real_client.post(f"/api/v1/images/{image['id']}/approve")
    real_client.post(
        "/api/v1/annotations",
        json={"image_id": image["id"], "class_id": 0, "class_name": "cone", "x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.3},
    )
    version = real_client.post(f"/api/v1/datasets/{dataset['id']}/versions", json={}).json()
    return project["id"], version["id"]


def test_roboflow_export_job_completes_and_uploads(
    connected_roboflow: TestClient, approved_version: tuple[str, str]
) -> None:
    _, version_id = approved_version

    resp = connected_roboflow.post(
        f"/api/v1/versions/{version_id}/export/roboflow",
        json={"workspace": "my-workspace", "project": "cones"},
    )
    assert resp.status_code == 202, resp.text
    job = resp.json()
    assert job["kind"] == "EXPORT"
    assert job["status"] == "COMPLETED"
    assert job["total_items"] == 1
    assert job["processed_items"] == 1
    assert job["uploaded_count"] == 1
    assert job["failed_count"] == 0
    assert job["failures"] == []

    progress = get_progress(job["id"])
    assert progress is not None
    assert progress.status == "COMPLETED"
    assert progress.current == 1


def test_roboflow_export_job_requires_connection_first(real_client: TestClient, unique_name: str) -> None:
    # `real_client` commits for real (unlike `client`'s rollback-per-test
    # transaction) — an earlier test in this file may have left Roboflow
    # connected, so disconnect explicitly rather than relying on run order.
    real_client.delete("/api/v1/integrations/roboflow")
    project_id = real_client.post("/api/v1/projects", json={"name": unique_name}).json()["id"]
    real_client.patch(
        f"/api/v1/projects/{project_id}", json={"class_config": [{"id": 0, "name": "cone"}]}
    )
    dataset_id = real_client.post(f"/api/v1/projects/{project_id}/datasets", json={"name": "d"}).json()["id"]
    image = real_client.post(
        f"/api/v1/datasets/{dataset_id}/images", files={"file": ("f.jpg", _jpeg_bytes(), "image/jpeg")}
    ).json()
    real_client.post(f"/api/v1/images/{image['id']}/approve")
    real_client.post(
        "/api/v1/annotations",
        json={"image_id": image["id"], "class_id": 0, "class_name": "cone", "x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.3},
    )
    version_id = real_client.post(f"/api/v1/datasets/{dataset_id}/versions", json={}).json()["id"]

    resp = real_client.post(
        f"/api/v1/versions/{version_id}/export/roboflow",
        json={"workspace": "ws", "project": "proj"},
    )
    assert resp.status_code == 400
    assert "not connected" in resp.json()["detail"].lower()


def test_latest_roboflow_import_job_returns_newest_for_project(
    connected_roboflow: TestClient, unique_name: str
) -> None:
    """Backs the "reattach to a still-running job after navigating away or
    reloading" UI behavior — DatasetsPage/ExportPage poll this on mount
    instead of relying on component state that a page change would just
    throw away."""
    project_id = connected_roboflow.post("/api/v1/projects", json={"name": unique_name}).json()["id"]

    first = connected_roboflow.post(
        f"/api/v1/projects/{project_id}/import/roboflow",
        json={"workspace": "my-workspace", "project": "cones", "version": 1},
    ).json()
    second = connected_roboflow.post(
        f"/api/v1/projects/{project_id}/import/roboflow",
        json={"workspace": "my-workspace", "project": "cones", "version": 1},
    ).json()
    assert first["id"] != second["id"]

    resp = connected_roboflow.get(
        f"/api/v1/integrations/roboflow/jobs/latest?kind=IMPORT&project_id={project_id}"
    )
    assert resp.status_code == 200
    assert resp.json()["id"] == second["id"]


def test_latest_roboflow_export_job_returns_newest_for_version(
    connected_roboflow: TestClient, approved_version: tuple[str, str]
) -> None:
    _, version_id = approved_version

    first = connected_roboflow.post(
        f"/api/v1/versions/{version_id}/export/roboflow",
        json={"workspace": "my-workspace", "project": "cones"},
    ).json()
    second = connected_roboflow.post(
        f"/api/v1/versions/{version_id}/export/roboflow",
        json={"workspace": "my-workspace", "project": "cones"},
    ).json()
    assert first["id"] != second["id"]

    resp = connected_roboflow.get(
        f"/api/v1/integrations/roboflow/jobs/latest?kind=EXPORT&dataset_version_id={version_id}"
    )
    assert resp.status_code == 200
    assert resp.json()["id"] == second["id"]


def test_latest_roboflow_job_none_found_returns_null(connected_roboflow: TestClient, unique_name: str) -> None:
    project_id = connected_roboflow.post("/api/v1/projects", json={"name": unique_name}).json()["id"]
    resp = connected_roboflow.get(
        f"/api/v1/integrations/roboflow/jobs/latest?kind=IMPORT&project_id={project_id}"
    )
    assert resp.status_code == 200
    assert resp.json() is None


def test_latest_roboflow_job_missing_filter_400(connected_roboflow: TestClient) -> None:
    assert connected_roboflow.get("/api/v1/integrations/roboflow/jobs/latest?kind=IMPORT").status_code == 400
    assert connected_roboflow.get("/api/v1/integrations/roboflow/jobs/latest?kind=EXPORT").status_code == 400


def test_roboflow_import_job_requires_connection_first(real_client: TestClient, unique_name: str) -> None:
    real_client.delete("/api/v1/integrations/roboflow")
    project_id = real_client.post("/api/v1/projects", json={"name": unique_name}).json()["id"]
    resp = real_client.post(
        f"/api/v1/projects/{project_id}/import/roboflow",
        json={"workspace": "ws", "project": "proj", "version": 1},
    )
    assert resp.status_code == 400
    assert "not connected" in resp.json()["detail"].lower()
