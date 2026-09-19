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
import re
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


# Raw (unversioned) pull path result set — two items on purpose, one
# already labeled and one not, so tests can exercise both "pull everything"
# and "unannotated_only" against the same fake project.
_RAW_SEARCH_ITEMS = [
    {"id": "raw-img-1", "name": "raw1.jpg", "annotations": {"count": 1, "classes": {"cone": 1}}},
    {"id": "raw-img-2", "name": "raw2.jpg", "annotations": {"count": 0, "classes": {}}},
]


class _FakeSearchResponse:
    """Stands in for the `requests.post(.../search)` response that
    `roboflow_search.rf_search_page` now inspects directly (the SDK's own
    `Project.search()` is bypassed)."""

    def __init__(self, items: list[dict], status_code: int = 200) -> None:
        self._items = items
        self.status_code = status_code
        self.text = "OK"

    def json(self) -> dict:
        return {"results": self._items}


def _fake_search_post(url: str, json: dict | None = None, timeout: int = 30) -> _FakeSearchResponse:
    offset = (json or {}).get("offset", 0)
    return _FakeSearchResponse([] if offset > 0 else list(_RAW_SEARCH_ITEMS))


def _image_details_payload(image_id: str) -> dict:
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


class _FakeImageDetailsResponse:
    """Stands in for the `requests.get(.../images/<id>)` response that
    `roboflow_import._rf_image_details` now inspects directly (the SDK's
    own `Project.image()` is bypassed — same reasoning as `rf_search_page`
    bypassing `Project.search()`: no timeout on the SDK's own call)."""

    def __init__(self, image_id: str, status_code: int = 200) -> None:
        self.status_code = status_code
        self._image_id = image_id

    def json(self) -> dict:
        return {"image": _image_details_payload(self._image_id)}


def _image_id_from_url(url: str) -> str:
    return url.split("/images/")[1].split("?")[0]


def _fake_get(url: str, params: dict | None = None, timeout: int = 30):
    """Default `requests.get` stand-in for the raw-pull tests below: an
    image-detail request (`.../images/<id>`, `api_key` sent via `params`
    now, not baked into `url`) returns the fake per-image payload, anything
    else (the CDN download) returns fake JPEG bytes."""
    if "/images/" in url:
        return _FakeImageDetailsResponse(_image_id_from_url(url))
    return _FakeHTTPResponse(_jpeg_bytes())


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
        # `rf_search_page` builds the /search URL from `rf_project.id`
        # (canonical "workspace/project"), mirroring the real SDK.
        self.id = f"my-workspace/{slug}"
        self.uploads: list[tuple[str, str | None, str, str | None, bool]] = []
        self.annotation_jobs: list[dict] = []
        self.saved_annotations: list[dict] = []

    def version(self, v: int) -> _FakeRoboflowVersion:
        return _FakeRoboflowVersion(v)

    def get_batches(self) -> dict:
        # The two static fake batches are what `test_list_roboflow_batches`
        # asserts against; batches from uploads this instance actually made
        # (tracked in `self.uploads`) are appended dynamically so
        # `_assign_annotating_review_job` can find the real batch a test's
        # export just created, by name.
        dynamic = [
            {"id": f"batch-id-{name}", "name": name, "images": 0}
            for name in dict.fromkeys(u[3] for u in self.uploads if u[3])
        ]
        return {
            "batches": [
                {"id": "batch-1", "name": "Batch One", "images": 5},
                {"id": "batch-2", "name": "Batch Two", "images": 3},
                *dynamic,
            ]
        }

    def create_annotation_job(
        self,
        *,
        name: str | None = None,
        batch_id: str | None = None,
        labeler_email: str | None = None,
        reviewer_email: str | None = None,
        instructions: str | None = None,
    ) -> dict:
        if not batch_id or not labeler_email or not reviewer_email:
            raise ValueError("batch_id, labeler_email, and reviewer_email are required")
        self.annotation_jobs.append(
            {"name": name, "batch_id": batch_id, "labeler_email": labeler_email, "reviewer_email": reviewer_email}
        )
        return {"id": "job-1"}

    def upload(
        self,
        *,
        image_path: str,
        annotation_path: str | None,
        annotation_labelmap: str,
        split: str,
        batch_name: str | None = None,
        is_prediction: bool = False,
        annotation_overwrite: bool = False,
    ) -> None:
        self.uploads.append((image_path, annotation_path, split, batch_name, is_prediction))

    def save_annotation(
        self,
        *,
        annotation_path: str,
        annotation_labelmap,
        image_id: str,
        job_name: str | None = None,
        is_prediction: bool = False,
        annotation_overwrite: bool = False,
    ) -> tuple[dict, float, int]:
        if image_id == "missing-on-roboflow":
            from roboflow.adapters.rfapi import AnnotationSaveError

            raise AnnotationSaveError("not found", status_code=404)
        self.saved_annotations.append(
            {
                "annotation_path": annotation_path,
                "annotation_labelmap": annotation_labelmap,
                "image_id": image_id,
                "job_name": job_name,
                "is_prediction": is_prediction,
                "annotation_overwrite": annotation_overwrite,
            }
        )
        # Real SDK shape: `Project.save_annotation()` returns
        # `(annotation, upload_time, upload_retry_attempts)` — see
        # `roboflow/core/project.py`.
        return {"success": True}, 0.0, 0

    # Raw (unversioned) pull path — no `.version()` here. The service no
    # longer calls `rf_project.search()` or `rf_project.image()`; it hits
    # /search and the per-image detail endpoint directly, so both are faked
    # via `_fake_search_post`/`_fake_get` (monkeypatched onto
    # `roboflow_import.requests.post`/`.get`) instead of a method here.


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


def test_image_model_has_roboflow_provenance_columns(real_db_session, unique_name: str) -> None:
    """New columns exist and round-trip — the foundation Task 3 (import)
    writes to and Task 5 (export) reads from."""
    from app.models.image import Image, ImageSourceType
    from app.models.project import Project
    from app.models.dataset import Dataset

    project = Project(name=unique_name, slug=unique_name, class_config=[{"id": 0, "name": "cone"}])
    real_db_session.add(project)
    real_db_session.flush()
    dataset = Dataset(project_id=project.id, name="ds")
    real_db_session.add(dataset)
    real_db_session.flush()

    image = Image(
        project_id=project.id,
        dataset_id=dataset.id,
        storage_key="k",
        original_filename="a.jpg",
        width=64,
        height=48,
        source_type=ImageSourceType.UPLOAD,
        roboflow_image_id="rf-abc123",
        roboflow_workspace="my-workspace",
        roboflow_project_slug="cones",
    )
    real_db_session.add(image)
    real_db_session.commit()
    real_db_session.refresh(image)

    assert image.roboflow_image_id == "rf-abc123"
    assert image.roboflow_workspace == "my-workspace"
    assert image.roboflow_project_slug == "cones"


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
    assert stats["approved_images"] == 0


def test_roboflow_import_job_images_only_skips_labels_versioned(
    connected_roboflow: TestClient, unique_name: str
) -> None:
    """`images_only: true` on a versioned pull still downloads the image but
    must not create the annotation the fake version's label file carries —
    the whole point is letting auto-annotate start from a clean slate."""
    project_id = connected_roboflow.post("/api/v1/projects", json={"name": unique_name}).json()["id"]

    resp = connected_roboflow.post(
        f"/api/v1/projects/{project_id}/import/roboflow",
        json={"workspace": "my-workspace", "project": "cones", "version": 1, "images_only": True},
    )
    assert resp.status_code == 202, resp.text
    job = resp.json()
    assert job["images_only"] is True
    assert job["status"] == "COMPLETED"
    assert job["total_items"] == 1

    dataset_id = job["result_dataset_id"]
    images = connected_roboflow.get(f"/api/v1/datasets/{dataset_id}/images").json()["items"]
    assert len(images) == 1
    annotations = connected_roboflow.get(f"/api/v1/images/{images[0]['id']}/annotations").json()
    assert annotations == []


def test_roboflow_import_job_raw_pull_when_no_version(
    connected_roboflow: TestClient, monkeypatch, unique_name: str
) -> None:
    """No `version` in the request — the project has nothing generated in
    Roboflow yet, so the job should fall back to `import_roboflow_raw_project`
    (raw `search()`/`image()` pull) instead of `.download()`."""
    import app.services.integrations.roboflow_import as roboflow_import_module

    monkeypatch.setattr(roboflow_import_module.requests, "get", _fake_get)
    monkeypatch.setattr(roboflow_import_module.requests, "post", _fake_search_post)

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
    # Both versioned and raw pulls leave images PENDING for review.
    assert stats["approved_images"] == 0

    project = connected_roboflow.get(f"/api/v1/projects/{project_id}").json()
    assert any(c["name"] == "cone" for c in project["class_config"])


def test_roboflow_raw_pull_import_persists_roboflow_image_id(
    connected_roboflow: TestClient, monkeypatch, unique_name: str
) -> None:
    """The raw-pull path already fetches each item's Roboflow id
    (`_rf_image_details(..., item["id"])`) — it must now also store it, so
    export can route these images through `save_annotation()` directly
    instead of re-uploading them (Task 5)."""
    import app.services.integrations.roboflow_import as roboflow_import_module

    monkeypatch.setattr(roboflow_import_module.requests, "get", _fake_get)
    monkeypatch.setattr(roboflow_import_module.requests, "post", _fake_search_post)

    project_id = connected_roboflow.post("/api/v1/projects", json={"name": unique_name}).json()["id"]

    resp = connected_roboflow.post(
        f"/api/v1/projects/{project_id}/import/roboflow",
        json={"workspace": "my-workspace", "project": "ground"},
    )
    assert resp.status_code == 202, resp.text
    job = resp.json()
    assert job["status"] == "COMPLETED"

    dataset_id = job["result_dataset_id"]
    images = connected_roboflow.get(f"/api/v1/datasets/{dataset_id}/images").json()["items"]
    assert len(images) == 2

    from app.db.session import SessionLocal
    from app.models.image import Image

    db = SessionLocal()
    try:
        rows = db.query(Image).filter(Image.dataset_id == dataset_id).all()
        ids = {r.roboflow_image_id for r in rows}
        assert ids == {"raw-img-1", "raw-img-2"}
        assert all(r.roboflow_workspace == "my-workspace" for r in rows)
        assert all(r.roboflow_project_slug == "ground" for r in rows)
    finally:
        db.close()


def test_roboflow_import_job_raw_pull_unannotated_only(
    connected_roboflow: TestClient, monkeypatch, unique_name: str
) -> None:
    """`unannotated_only: true` narrows the raw pull to the one fake item
    with zero existing Roboflow annotations (raw-img-2) — raw-img-1, which
    already has a box, must be skipped entirely."""
    import app.services.integrations.roboflow_import as roboflow_import_module

    monkeypatch.setattr(roboflow_import_module.requests, "get", _fake_get)
    monkeypatch.setattr(roboflow_import_module.requests, "post", _fake_search_post)

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


def test_roboflow_import_job_raw_pull_images_only_skips_labels(
    connected_roboflow: TestClient, monkeypatch, unique_name: str
) -> None:
    """`images_only: true` on a raw pull brings in every image (unlike
    `unannotated_only`, nothing is filtered out) but must not create the
    annotation that raw-img-1's fake `annotation.boxes` carries."""
    import app.services.integrations.roboflow_import as roboflow_import_module

    monkeypatch.setattr(roboflow_import_module.requests, "get", _fake_get)
    monkeypatch.setattr(roboflow_import_module.requests, "post", _fake_search_post)

    project_id = connected_roboflow.post("/api/v1/projects", json={"name": unique_name}).json()["id"]

    resp = connected_roboflow.post(
        f"/api/v1/projects/{project_id}/import/roboflow",
        json={"workspace": "my-workspace", "project": "ground", "images_only": True},
    )
    assert resp.status_code == 202, resp.text
    job = resp.json()
    assert job["images_only"] is True
    assert job["status"] == "COMPLETED"
    assert job["total_items"] == 2

    dataset_id = job["result_dataset_id"]
    images = connected_roboflow.get(f"/api/v1/datasets/{dataset_id}/images").json()["items"]
    assert len(images) == 2
    for image in images:
        annotations = connected_roboflow.get(f"/api/v1/images/{image['id']}/annotations").json()
        assert annotations == []


def test_list_roboflow_batches(connected_roboflow: TestClient) -> None:
    resp = connected_roboflow.get("/api/v1/integrations/roboflow/projects/my-workspace/ground/batches")
    assert resp.status_code == 200, resp.text
    batches = resp.json()
    assert batches == [
        {"id": "batch-1", "name": "Batch One", "image_count": 5},
        {"id": "batch-2", "name": "Batch Two", "image_count": 3},
    ]


def test_rf_search_page_forwards_batch_id_in_payload(monkeypatch) -> None:
    """`batch_id`, when passed, must reach the /search payload as Roboflow's
    own `search(batch=True, batch_id=...)` would send it — this is the
    plumbing `import_roboflow_raw_project`'s `batch_id` param relies on to
    actually narrow the pull server-side."""
    import app.services.integrations.roboflow_search as mod

    captured: dict = {}

    def fake_post(url, json=None, timeout=30):
        captured.update(json or {})
        return _SeqResp(200, {"results": []})

    monkeypatch.setattr(mod.requests, "post", fake_post)

    rf_project = type("P", (), {"id": "ws/proj"})()
    mod.rf_search_page(rf_project, "key", offset=0, limit=100, fields=["id"], batch_id="batch-1")

    assert captured["batch"] is True
    assert captured["batch_id"] == "batch-1"


def test_roboflow_import_job_raw_pull_by_batch_id(
    connected_roboflow: TestClient, monkeypatch, unique_name: str
) -> None:
    """`batch_id` on the import request is stored on the job and forwarded
    to every /search page — the actual filtering happens server-side on
    Roboflow, so this only verifies the plumbing through the job."""
    import app.services.integrations.roboflow_import as roboflow_import_module

    captured_payloads: list[dict] = []

    def _fake_post(url: str, json: dict | None = None, timeout: int = 30) -> _FakeSearchResponse:
        captured_payloads.append(json or {})
        return _fake_search_post(url, json=json, timeout=timeout)

    monkeypatch.setattr(roboflow_import_module.requests, "get", _fake_get)
    monkeypatch.setattr(roboflow_import_module.requests, "post", _fake_post)

    project_id = connected_roboflow.post("/api/v1/projects", json={"name": unique_name}).json()["id"]

    resp = connected_roboflow.post(
        f"/api/v1/projects/{project_id}/import/roboflow",
        json={"workspace": "my-workspace", "project": "ground", "batch_id": "batch-1"},
    )
    assert resp.status_code == 202, resp.text
    job = resp.json()
    assert job["batch_id"] == "batch-1"
    assert job["status"] == "COMPLETED"

    assert captured_payloads, "expected at least one /search call"
    assert all(p.get("batch") is True and p.get("batch_id") == "batch-1" for p in captured_payloads)


def test_rf_search_error_body_surfaces_as_runtime_error(
    connected_roboflow: TestClient, real_db_session, monkeypatch, unique_name: str
) -> None:
    """Regression: the Roboflow SDK's `Project.search()` ends with a bare
    `data.json()["results"]`, so an `{"error": ...}` body from the /search
    endpoint used to blow up as an opaque `KeyError: 'results'` from inside
    the SDK. `rf_search_page` must instead raise a `RuntimeError` that
    carries the real HTTP status and response body."""
    import uuid as _uuid

    import app.services.integrations.roboflow_import as roboflow_import_module
    from app.services.integrations.roboflow_import import import_roboflow_raw_project

    class _ErrResp:
        status_code = 401
        text = '{"error": "no search access"}'

        def json(self) -> dict:
            return {"error": "This API key does not have search access."}

    monkeypatch.setattr(
        roboflow_import_module.requests, "post", lambda url, json=None, timeout=30: _ErrResp()
    )

    project_id = connected_roboflow.post("/api/v1/projects", json={"name": unique_name}).json()["id"]

    with pytest.raises(RuntimeError) as excinfo:
        import_roboflow_raw_project(
            real_db_session,
            project_id=_uuid.UUID(project_id),
            workspace="my-workspace",
            project_slug="ground",
            dataset_name=None,
        )
    msg = str(excinfo.value)
    assert "search access" in msg
    assert "HTTP 401" in msg


class _SeqResp:
    """Minimal `requests.Response` stand-in for the retry tests."""

    def __init__(self, status_code: int, payload: dict, text: str = "x") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> dict:
        return self._payload


def test_rf_search_retries_transient_5xx_then_succeeds(monkeypatch) -> None:
    """A transient 5xx from /search (observed live: Roboflow returned bare
    HTTP 500s for a few minutes) is retried with backoff, not fatal — the
    page load recovers as soon as Roboflow returns 200 again."""
    import app.services.integrations.roboflow_search as mod

    seq = [
        _SeqResp(500, {"error": "An error occurred with this request"}),
        _SeqResp(503, {"error": "upstream"}),
        _SeqResp(200, {"results": [{"id": "a"}]}),
    ]
    calls = {"n": 0}

    def fake_post(url, json=None, timeout=30):
        r = seq[calls["n"]]
        calls["n"] += 1
        return r

    slept: list[float] = []
    monkeypatch.setattr(mod.requests, "post", fake_post)
    monkeypatch.setattr(mod.time, "sleep", lambda s: slept.append(s))

    rf_project = type("P", (), {"id": "ws/proj"})()
    out = mod.rf_search_page(rf_project, "key", offset=0, limit=100, fields=["id"])

    assert out == [{"id": "a"}]
    assert calls["n"] == 3
    assert slept == [1.0, 2.0]  # exponential backoff between the three attempts


def test_rf_search_persistent_5xx_raises_with_retry_hint(monkeypatch) -> None:
    """When every attempt 5xxs, the raised error keeps the real status/body
    and tells the user it's a transient Roboflow-side problem to retry."""
    import app.services.integrations.roboflow_search as mod

    calls = {"n": 0}

    def fake_post(url, json=None, timeout=30):
        calls["n"] += 1
        return _SeqResp(500, {"error": "An error occurred with this request, please try again."})

    monkeypatch.setattr(mod.requests, "post", fake_post)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    rf_project = type("P", (), {"id": "ws/proj"})()
    with pytest.raises(RuntimeError) as excinfo:
        mod.rf_search_page(rf_project, "key", offset=0, limit=100, fields=["id"])

    msg = str(excinfo.value)
    assert calls["n"] == mod._SEARCH_MAX_ATTEMPTS
    assert "HTTP 500" in msg
    assert "temporary Roboflow-side issue" in msg


def test_rf_search_connection_error_is_retried(monkeypatch) -> None:
    """A `requests` connection/timeout error is retried the same way, and
    the final failure is a clear RuntimeError rather than a bare socket
    exception bubbling out of the job."""
    import app.services.integrations.roboflow_search as mod

    calls = {"n": 0}

    def fake_post(url, json=None, timeout=30):
        calls["n"] += 1
        if calls["n"] < 3:
            raise mod.requests.ConnectionError("connection reset")
        return _SeqResp(200, {"results": []})

    monkeypatch.setattr(mod.requests, "post", fake_post)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    rf_project = type("P", (), {"id": "ws/proj"})()
    out = mod.rf_search_page(rf_project, "key", offset=0, limit=100, fields=["id"])
    assert out == []
    assert calls["n"] == 3


def test_rf_search_dns_failure_raises_local_network_hint(monkeypatch) -> None:
    """A connection error that never recovers (observed live: DNS resolution
    failure for api.roboflow.com) must NOT tell the user it's a "temporary
    Roboflow-side issue" — that's actively misleading when the request never
    reached Roboflow at all. It should point at this machine's network/DNS
    instead."""
    import app.services.integrations.roboflow_search as mod

    def fake_post(url, json=None, timeout=30):
        raise mod.requests.ConnectionError(
            "HTTPSConnectionPool(host='api.roboflow.com', port=443): Max retries exceeded "
            "(Caused by NameResolutionError: Failed to resolve 'api.roboflow.com' "
            "([Errno 11001] getaddrinfo failed))"
        )

    monkeypatch.setattr(mod.requests, "post", fake_post)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    rf_project = type("P", (), {"id": "ws/proj"})()
    with pytest.raises(RuntimeError) as excinfo:
        mod.rf_search_page(rf_project, "key", offset=0, limit=100, fields=["id"])

    msg = str(excinfo.value)
    assert "temporary Roboflow-side issue" not in msg
    assert "network" in msg.lower() or "dns" in msg.lower()


def test_rf_image_details_retries_bad_json_then_succeeds(monkeypatch) -> None:
    """Regression: `Project.image()` (the SDK's per-image detail fetch) ends
    with a bare `requests.get(url).json()` — a transient blip returning a
    non-JSON body used to blow up the whole import as an opaque
    `json.JSONDecodeError: Expecting value: line 2 column 1 (char 1)`
    (observed live, 35/40 images in). `_rf_image_details` must retry that
    failure and recover once the request succeeds again."""
    import app.services.integrations.roboflow_import as mod

    calls = {"n": 0}

    class _BadJSONResp:
        status_code = 200

        def json(self) -> dict:
            raise ValueError("Expecting value: line 2 column 1 (char 1)")

    def fake_get(url, params=None, timeout=30):
        calls["n"] += 1
        if calls["n"] < 3:
            return _BadJSONResp()
        return _SeqResp(200, {"image": {"id": "img-1", "urls": {"original": "http://x/img.jpg"}}})

    monkeypatch.setattr(mod.requests, "get", fake_get)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    rf_project = type("P", (), {"id": "ws/proj"})()
    out = mod._rf_image_details(rf_project, "key", "img-1")
    assert out == {"id": "img-1", "urls": {"original": "http://x/img.jpg"}}
    assert calls["n"] == 3


def test_rf_image_details_persistent_failure_returns_none(monkeypatch) -> None:
    """Once retries are exhausted, `_rf_image_details` returns `None`
    instead of raising, so the caller can skip just this one image (same
    idiom as an unreadable image file) rather than aborting the whole
    multi-image import over one persistently bad item."""
    import app.services.integrations.roboflow_import as mod

    def fake_get(url, params=None, timeout=30):
        return _SeqResp(200, {"error": "Image not found"})

    monkeypatch.setattr(mod.requests, "get", fake_get)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    rf_project = type("P", (), {"id": "ws/proj"})()
    out = mod._rf_image_details(rf_project, "key", "img-1")
    assert out is None


def test_rf_image_details_passes_bounded_timeout(monkeypatch) -> None:
    """Regression: the SDK's `Project.image()` ends with a bare
    `requests.get(url).json()` — no timeout at all, so a hung connection
    (as opposed to a 5xx/429, which at least raises) blocks forever. Under
    the raw pull's concurrent per-image fetcher (`_run_windowed`), that
    freezes the whole import's progress AND its cancel check (only
    re-polled once a fetch completes), so Cancel does nothing. Every
    `_rf_image_details` request must carry an explicit timeout so a hang
    always surfaces as a retryable error instead."""
    import app.services.integrations.roboflow_import as mod

    captured: dict = {}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return _SeqResp(200, {"image": {"id": "img-1"}})

    monkeypatch.setattr(mod.requests, "get", fake_get)

    rf_project = type("P", (), {"id": "ws/proj"})()
    out = mod._rf_image_details(rf_project, "super-secret-key", "img-1")

    assert out == {"id": "img-1"}
    assert captured["timeout"] == mod._IMAGE_DETAIL_TIMEOUT_S
    assert captured["timeout"] is not None
    # The key must travel as a query param, not be baked into `url` itself —
    # otherwise it ends up in any log line or error message that includes
    # the URL (see roboflow_import.py's `_rf_image_details` docstring).
    assert "super-secret-key" not in captured["url"]
    assert captured["params"] == {"api_key": "super-secret-key"}


def test_download_version_dataset_retries_transient_failure_then_succeeds(monkeypatch, tmp_path) -> None:
    """Regression: `Version.download()` (the versioned pull's one big
    blocking call) raises a bare `RuntimeError` on any transient 5xx/429
    from Roboflow's export-status endpoint, with no retry of its own — one
    blip used to abort the whole (often multi-minute) versioned import.
    `_download_version_dataset` must retry the whole call and succeed once
    Roboflow recovers."""
    import app.services.integrations.roboflow_import as mod

    calls = {"n": 0}
    location = str(tmp_path / "download")

    class _FakeVersion:
        def download(self, fmt: str, location: str):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("{'error': 'An error occurred with this request'}")
            Path(location).mkdir(parents=True)
            (Path(location) / "data.yaml").write_text("names: ['cone']\n", encoding="utf-8")
            return location

    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    out = mod._download_version_dataset(_FakeVersion(), "yolov8", location)
    assert out == location
    assert calls["n"] == 3
    assert (Path(location) / "data.yaml").exists()


def test_download_version_dataset_clears_partial_dir_before_retry(monkeypatch, tmp_path) -> None:
    """`Version.download()` treats an already-existing `location` as
    "already downloaded" and returns without downloading anything —
    without clearing `location` between attempts, a retry after a failure
    that left partial files behind would silently resurrect that partial,
    corrupt dataset instead of forcing a real re-download."""
    import app.services.integrations.roboflow_import as mod

    calls = {"n": 0}
    location = str(tmp_path / "download")

    class _FakeVersion:
        def download(self, fmt: str, location: str):
            calls["n"] += 1
            if calls["n"] < 2:
                # Simulate dying mid-extract: partial files exist even
                # though this attempt is about to fail.
                Path(location).mkdir(parents=True, exist_ok=True)
                (Path(location) / "partial.txt").write_text("incomplete", encoding="utf-8")
                raise RuntimeError("transient")
            # A real download would refuse to proceed into a non-empty
            # directory from a prior attempt — asserting it's clean here
            # proves `_download_version_dataset` cleared it first.
            assert not Path(location).exists() or list(Path(location).iterdir()) == []
            Path(location).mkdir(parents=True, exist_ok=True)
            (Path(location) / "data.yaml").write_text("names: ['cone']\n", encoding="utf-8")
            return location

    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    mod._download_version_dataset(_FakeVersion(), "yolov8", location)
    assert calls["n"] == 2


def test_download_version_dataset_persistent_failure_raises(monkeypatch, tmp_path) -> None:
    import app.services.integrations.roboflow_import as mod

    calls = {"n": 0}
    location = str(tmp_path / "download")

    class _FakeVersion:
        def download(self, fmt: str, location: str):
            calls["n"] += 1
            raise RuntimeError("permanent failure")

    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    with pytest.raises(RuntimeError, match="permanent failure"):
        mod._download_version_dataset(_FakeVersion(), "yolov8", location)
    assert calls["n"] == mod._VERSION_DOWNLOAD_MAX_ATTEMPTS


def test_roboflow_import_job_skips_one_bad_image_instead_of_failing_job(
    connected_roboflow: TestClient, monkeypatch, unique_name: str
) -> None:
    """A single image whose detail fetch keeps failing (the same transient
    blip `_rf_image_details` retries) must not fail the whole job — the
    other, healthy image still imports and the job completes."""
    import app.services.integrations.roboflow_import as roboflow_import_module

    def _flaky_get(url: str, params: dict | None = None, timeout: int = 30):
        if "/images/" in url:
            image_id = _image_id_from_url(url)
            if image_id == "raw-img-1":
                raise ValueError("Expecting value: line 2 column 1 (char 1)")
            return _FakeImageDetailsResponse(image_id)
        return _FakeHTTPResponse(_jpeg_bytes())

    monkeypatch.setattr(roboflow_import_module.requests, "get", _flaky_get)
    monkeypatch.setattr(roboflow_import_module.requests, "post", _fake_search_post)
    monkeypatch.setattr(roboflow_import_module.time, "sleep", lambda s: None)

    project_id = connected_roboflow.post("/api/v1/projects", json={"name": unique_name}).json()["id"]

    resp = connected_roboflow.post(
        f"/api/v1/projects/{project_id}/import/roboflow",
        json={"workspace": "my-workspace", "project": "ground"},
    )
    assert resp.status_code == 202, resp.text
    job = resp.json()
    assert job["status"] == "COMPLETED"
    assert job["total_items"] == 2

    dataset_id = job["result_dataset_id"]
    images = connected_roboflow.get(f"/api/v1/datasets/{dataset_id}/images").json()["items"]
    assert len(images) == 1
    assert images[0]["original_filename"] == "raw2.jpg"


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

    monkeypatch.setattr(roboflow_import_module.requests, "get", _fake_get)
    monkeypatch.setattr(roboflow_import_module.requests, "post", _fake_search_post)

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

    monkeypatch.setattr(roboflow_import_module.requests, "get", _fake_get)
    monkeypatch.setattr(roboflow_import_module.requests, "post", _fake_search_post)

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


def test_roboflow_export_known_image_id_updates_annotation_without_reupload(
    connected_roboflow: TestClient, unique_name: str
) -> None:
    """An image this app already knows the Roboflow id for (imported from
    the same workspace/project being exported to) must be routed straight
    through save_annotation — Project.upload() must never be called for
    it at all. Builds the exportable version through the exact same real
    API calls as the `approved_version` fixture above (`real_client.post`
    for project/dataset/image/approve/annotation/version), then sets
    `roboflow_image_id`/`roboflow_workspace`/`roboflow_project_slug` on
    the `Image` row directly afterward — there is no API for setting those
    outside the real Roboflow import flow, which this test deliberately
    isn't invoking (it only needs the *result* of having imported, not the
    import itself — Task 3 already covers the import path writing these)."""
    from app.db.session import SessionLocal
    from app.models.image import Image

    project = connected_roboflow.post("/api/v1/projects", json={"name": unique_name}).json()
    connected_roboflow.patch(
        f"/api/v1/projects/{project['id']}", json={"class_config": [{"id": 0, "name": "cone"}]}
    )
    dataset = connected_roboflow.post(f"/api/v1/projects/{project['id']}/datasets", json={"name": "d"}).json()
    image = connected_roboflow.post(
        f"/api/v1/datasets/{dataset['id']}/images", files={"file": ("f.jpg", _jpeg_bytes(), "image/jpeg")}
    ).json()
    connected_roboflow.post(f"/api/v1/images/{image['id']}/approve")
    connected_roboflow.post(
        "/api/v1/annotations",
        json={
            "image_id": image["id"],
            "class_id": 0,
            "class_name": "cone",
            "x1": 0.1,
            "y1": 0.1,
            "x2": 0.3,
            "y2": 0.3,
        },
    )

    db = SessionLocal()
    try:
        db.query(Image).filter(Image.id == image["id"]).update(
            {
                "roboflow_image_id": "known-rf-image-id",
                "roboflow_workspace": "my-workspace",
                "roboflow_project_slug": "cones",
            }
        )
        db.commit()
    finally:
        db.close()

    version = connected_roboflow.post(f"/api/v1/datasets/{dataset['id']}/versions", json={}).json()

    resp = connected_roboflow.post(
        f"/api/v1/versions/{version['id']}/export/roboflow",
        json={"workspace": "my-workspace", "project": "cones"},
    )
    assert resp.status_code == 202, resp.text
    job = resp.json()
    assert job["status"] == "COMPLETED", job
    assert job["new_images_count"] == 0
    assert job["annotations_updated_count"] == 1


def test_roboflow_export_names_batch_after_app_and_dataset_version(
    connected_roboflow: TestClient, approved_version: tuple[str, str], monkeypatch
) -> None:
    """Regression: pushing to Roboflow left every export under the SDK's own
    default batch name ("Pip Package Upload") since `push_version_to_roboflow`
    never passed `batch_name` — indistinguishable in Roboflow's UI from any
    other tool's uploads. Exports must be grouped under a batch that
    identifies this app and which dataset version was pushed — and the name
    must be Roboflow-safe (`^[a-z0-9_-]{1,64}$`), or the upload is dropped
    server-side and the image never reaches the Annotate tab."""
    _, version_id = approved_version

    captured: list[dict] = []
    original_upload = _FakeRoboflowProject.upload

    def _spy_upload(self, **kwargs):
        captured.append(kwargs)
        return original_upload(self, **kwargs)

    monkeypatch.setattr(_FakeRoboflowProject, "upload", _spy_upload)

    resp = connected_roboflow.post(
        f"/api/v1/versions/{version_id}/export/roboflow",
        json={"workspace": "my-workspace", "project": "cones"},
    )
    assert resp.status_code == 202, resp.text
    assert resp.json()["status"] == "COMPLETED"

    assert captured
    # `approved_version`'s dataset is named "d"; its first version is v1.
    assert all(c["batch_name"] == "autolabelflow-d-v1" for c in captured)
    assert all(re.fullmatch(r"[a-z0-9_-]{1,64}", c["batch_name"]) for c in captured)


def test_roboflow_export_uploads_annotations_as_predictions_not_ground_truth(
    connected_roboflow: TestClient, approved_version: tuple[str, str], monkeypatch
) -> None:
    """Regression: pushing to Roboflow uploaded annotations as ground truth,
    which Roboflow auto-confirms — every pushed image landed straight in the
    "Dataset" column of the Annotate board, fully "Annotated", with no
    chance for a human to review labels that came from our pipeline, not a
    person. Uploads must set `is_prediction=True` so Roboflow queues the
    image for review under "Annotating" instead."""
    _, version_id = approved_version

    captured: list[dict] = []
    original_upload = _FakeRoboflowProject.upload

    def _spy_upload(self, **kwargs):
        captured.append(kwargs)
        return original_upload(self, **kwargs)

    monkeypatch.setattr(_FakeRoboflowProject, "upload", _spy_upload)

    resp = connected_roboflow.post(
        f"/api/v1/versions/{version_id}/export/roboflow",
        json={"workspace": "my-workspace", "project": "cones"},
    )
    assert resp.status_code == 202, resp.text
    assert resp.json()["status"] == "COMPLETED"

    assert captured
    assert all(c["is_prediction"] is True for c in captured)


def test_roboflow_export_upload_target_dataset_pushes_ground_truth(
    connected_roboflow: TestClient, approved_version: tuple[str, str], monkeypatch
) -> None:
    """`upload_target: "dataset"` opts out of the "Annotating" review queue
    (the default) and pushes labels as ground truth instead, so Roboflow
    auto-confirms them straight into the Dataset column."""
    _, version_id = approved_version

    captured: list[dict] = []
    original_upload = _FakeRoboflowProject.upload

    def _spy_upload(self, **kwargs):
        captured.append(kwargs)
        return original_upload(self, **kwargs)

    monkeypatch.setattr(_FakeRoboflowProject, "upload", _spy_upload)

    resp = connected_roboflow.post(
        f"/api/v1/versions/{version_id}/export/roboflow",
        json={"workspace": "my-workspace", "project": "cones", "upload_target": "DATASET"},
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "COMPLETED"
    assert body["upload_target"] == "DATASET"

    assert captured
    assert any(c["annotation_path"] is not None for c in captured)
    assert all(c["is_prediction"] is False for c in captured)


def test_roboflow_export_upload_target_unannotated_strips_labels(
    connected_roboflow: TestClient, approved_version: tuple[str, str], monkeypatch
) -> None:
    """`upload_target: "unannotated"` pushes every image with no annotation
    at all, even though the version being exported has approved labels —
    the point is landing images in Roboflow's unannotated bucket regardless
    of what's already labeled locally."""
    _, version_id = approved_version

    captured: list[dict] = []
    original_upload = _FakeRoboflowProject.upload

    def _spy_upload(self, **kwargs):
        captured.append(kwargs)
        return original_upload(self, **kwargs)

    monkeypatch.setattr(_FakeRoboflowProject, "upload", _spy_upload)

    resp = connected_roboflow.post(
        f"/api/v1/versions/{version_id}/export/roboflow",
        json={"workspace": "my-workspace", "project": "cones", "upload_target": "UNANNOTATED"},
    )
    assert resp.status_code == 202, resp.text
    assert resp.json()["status"] == "COMPLETED"

    assert captured
    assert all(c["annotation_path"] is None for c in captured)


def test_roboflow_export_upload_target_defaults_to_annotating(
    connected_roboflow: TestClient, approved_version: tuple[str, str]
) -> None:
    """No `upload_target` in the request must keep the historical default
    ("annotating") on the job row, not break existing callers."""
    _, version_id = approved_version

    resp = connected_roboflow.post(
        f"/api/v1/versions/{version_id}/export/roboflow",
        json={"workspace": "my-workspace", "project": "cones"},
    )
    assert resp.status_code == 202, resp.text
    assert resp.json()["upload_target"] == "ANNOTATING"


def test_roboflow_export_annotating_without_labeler_email_notes_it_non_fatally(
    connected_roboflow: TestClient, approved_version: tuple[str, str]
) -> None:
    """Regression: uploading with `is_prediction=True` alone only gets an
    image as far as Roboflow's Unassigned column — actually moving it to
    Annotating needs a second API call (create_annotation_job) that
    requires a labeler email. With none configured, the export must still
    COMPLETE (images did upload) but say why they're not in Annotating,
    rather than silently leaving the user to wonder."""
    _, version_id = approved_version

    resp = connected_roboflow.post(
        f"/api/v1/versions/{version_id}/export/roboflow",
        json={"workspace": "my-workspace", "project": "cones"},
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "COMPLETED"
    assert "labeler" in body["error"].lower()


def test_roboflow_export_annotating_creates_review_job_when_labeler_email_set(
    connected_roboflow: TestClient, approved_version: tuple[str, str], monkeypatch
) -> None:
    """With a default labeler email configured (Settings -> Roboflow), an
    ANNOTATING-target export must file a real Roboflow annotation job for
    the batch it just pushed, assigning the same email as both labeler and
    reviewer (this app has no UI for picking someone else to review its
    own auto-generated predictions) — and the job row must show the
    informational summary, not an error (a clean review job filing adds
    nothing further to it)."""
    resp = connected_roboflow.post(
        "/api/v1/integrations/roboflow",
        json={"api_key": "good-key", "default_labeler_email": "reviewer@example.com"},
    )
    assert resp.status_code == 200, resp.text

    _, version_id = approved_version

    captured: list[dict] = []
    original_create = _FakeRoboflowProject.create_annotation_job

    def _spy_create(self, **kwargs):
        captured.append(kwargs)
        return original_create(self, **kwargs)

    monkeypatch.setattr(_FakeRoboflowProject, "create_annotation_job", _spy_create)

    resp = connected_roboflow.post(
        f"/api/v1/versions/{version_id}/export/roboflow",
        json={"workspace": "my-workspace", "project": "cones", "batch_name": "review-me"},
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "COMPLETED"
    assert body["error"] == "1 new image uploaded."

    assert captured == [
        {
            "name": "review-me",
            "batch_id": "batch-id-review-me",
            "labeler_email": "reviewer@example.com",
            "reviewer_email": "reviewer@example.com",
        }
    ]


def test_roboflow_export_dataset_target_never_creates_annotation_job(
    connected_roboflow: TestClient, approved_version: tuple[str, str], monkeypatch
) -> None:
    """`DATASET` (ground truth) images already land in the right place on
    upload alone — no review job should ever be created for that target,
    labeler email configured or not."""
    resp = connected_roboflow.post(
        "/api/v1/integrations/roboflow",
        json={"api_key": "good-key", "default_labeler_email": "reviewer@example.com"},
    )
    assert resp.status_code == 200, resp.text

    _, version_id = approved_version

    created: list[dict] = []
    original_create = _FakeRoboflowProject.create_annotation_job

    def _spy_create(self, **kwargs):
        created.append(kwargs)
        return original_create(self, **kwargs)

    monkeypatch.setattr(_FakeRoboflowProject, "create_annotation_job", _spy_create)

    resp = connected_roboflow.post(
        f"/api/v1/versions/{version_id}/export/roboflow",
        json={"workspace": "my-workspace", "project": "cones", "upload_target": "DATASET"},
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "COMPLETED"
    assert body["error"] == "1 new image uploaded."
    assert created == []


def test_roboflow_connect_preserves_labeler_email_on_reconnect_without_it(
    connected_roboflow: TestClient,
) -> None:
    """Re-verifying the API key (e.g. Settings' "Connect" form re-submitted
    without touching the labeler-email field) must not silently wipe out an
    already-stored default labeler email — only an explicit new value
    should replace it."""
    resp = connected_roboflow.post(
        "/api/v1/integrations/roboflow",
        json={"api_key": "good-key", "default_labeler_email": "reviewer@example.com"},
    )
    assert resp.status_code == 200, resp.text

    resp = connected_roboflow.post("/api/v1/integrations/roboflow", json={"api_key": "good-key"})
    assert resp.status_code == 200, resp.text

    from app.db.session import SessionLocal
    from app.models.integration import Integration, IntegrationProvider
    from sqlalchemy import select

    db = SessionLocal()
    try:
        row = db.scalar(select(Integration).where(Integration.provider == IntegrationProvider.ROBOFLOW.value))
        assert row.config.get("default_labeler_email") == "reviewer@example.com"
    finally:
        db.close()


def test_roboflow_export_uses_custom_batch_name_when_given(
    connected_roboflow: TestClient, approved_version: tuple[str, str], monkeypatch
) -> None:
    """Regression: exporting always pushed into whatever existing project the
    dropdown had selected under an auto-generated batch label, so re-runs
    into a project that already carried annotations from a prior push/import
    were indistinguishable from those older uploads. A user-supplied
    `batch_name` must be used (Roboflow-sanitized) instead of the
    auto-generated `AutoLabelFlow-{dataset}-v{n}` one, so a push can be
    labeled/grouped however the user wants without needing a whole separate
    Roboflow project."""
    _, version_id = approved_version

    captured: list[dict] = []
    original_upload = _FakeRoboflowProject.upload

    def _spy_upload(self, **kwargs):
        captured.append(kwargs)
        return original_upload(self, **kwargs)

    monkeypatch.setattr(_FakeRoboflowProject, "upload", _spy_upload)

    resp = connected_roboflow.post(
        f"/api/v1/versions/{version_id}/export/roboflow",
        json={"workspace": "my-workspace", "project": "cones", "batch_name": "My Cool Batch!"},
    )
    assert resp.status_code == 202, resp.text
    assert resp.json()["status"] == "COMPLETED"

    assert captured
    assert all(c["batch_name"] == "my-cool-batch" for c in captured)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("AutoLabelFlow-d-v1", "autolabelflow-d-v1"),
        ("AutoLabelFlow (My Cones Set-v3)", "autolabelflow-my-cones-set-v3"),
        ("  spaces & (parens!) ", "spaces-parens"),
        ("A" * 80, "a" * 64),
        ("////", "autolabelflow"),
        ("日本語データ", "autolabelflow"),
    ],
)
def test_sanitize_batch_name_matches_roboflow_rule(raw: str, expected: str) -> None:
    from app.services.integrations.roboflow_export import _sanitize_batch_name

    out = _sanitize_batch_name(raw)
    assert out == expected
    assert re.fullmatch(r"[a-z0-9_-]{1,64}", out)


# --- upload retry / fail-fast on a Roboflow-side 5xx -----------------------
#
# Roboflow's /dataset/{project}/upload endpoint intermittently answers with
# a bare "500 Server Error" page (Google Frontend, no JSON) — and, when a
# workspace is out of monthly upload quota or on an expired plan, does so
# for *every* image. Without this handling a multi-thousand-image push
# grinds for hours, uploads nothing, and leaves the job "RUNNING" with a
# useless `f.jpg: <Response [500]>` per row.


def _upload_error(status_code: int):
    from roboflow.adapters.rfapi import ImageUploadError

    return ImageUploadError(f"<Response [{status_code}]>", status_code=status_code)


def test_upload_one_image_retries_transient_5xx_then_succeeds(monkeypatch) -> None:
    from app.services.integrations import roboflow_export as mod

    slept: list[float] = []
    monkeypatch.setattr(mod.time, "sleep", lambda s: slept.append(s))

    calls = {"n": 0}

    class _Proj:
        def upload(self, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise _upload_error(500)

    mod._upload_one_image(_Proj(), has_annotation=False, image_path="a.jpg")

    assert calls["n"] == 3
    assert slept == [1.0, 2.0]  # backoff between attempts 1->2 and 2->3


def test_upload_one_image_gives_up_after_max_attempts(monkeypatch) -> None:
    from roboflow.adapters.rfapi import ImageUploadError

    from app.services.integrations import roboflow_export as mod

    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    class _Proj:
        def upload(self, **kwargs):
            raise _upload_error(503)

    with pytest.raises(ImageUploadError):
        mod._upload_one_image(_Proj(), has_annotation=False, image_path="a.jpg")


def test_upload_one_image_does_not_retry_4xx(monkeypatch) -> None:
    from roboflow.adapters.rfapi import ImageUploadError

    from app.services.integrations import roboflow_export as mod

    monkeypatch.setattr(mod.time, "sleep", lambda s: pytest.fail("should not back off on a 4xx"))

    calls = {"n": 0}

    class _Proj:
        def upload(self, **kwargs):
            calls["n"] += 1
            raise _upload_error(400)

    with pytest.raises(ImageUploadError):
        mod._upload_one_image(_Proj(), has_annotation=False, image_path="a.jpg")
    assert calls["n"] == 1


def test_upload_one_image_retries_bare_connection_error(monkeypatch) -> None:
    """Regression: a plain `requests.ConnectionError`/timeout during
    `project.upload()` has no `status_code` at all, so
    `getattr(exc, "status_code", None)` is `None` — which isn't in
    `_UPLOAD_RETRY_STATUSES`, so this used to be treated as non-transient
    and fail the image on the first attempt instead of retrying."""
    from app.services.integrations import roboflow_export as mod

    slept: list[float] = []
    monkeypatch.setattr(mod.time, "sleep", lambda s: slept.append(s))

    calls = {"n": 0}

    class _Proj:
        def upload(self, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise mod.requests.ConnectionError("connection reset")

    mod._upload_one_image(_Proj(), has_annotation=False, image_path="a.jpg")

    assert calls["n"] == 3
    assert slept == [1.0, 2.0]


def test_fail_fast_message_points_at_quota_for_all_5xx() -> None:
    from app.services.integrations.roboflow_export import _fail_fast_message

    msg = _fail_fast_message([500, 500, 502, 500, 503], ["a.jpg: HTTP 500: x"] * 5)
    assert "quota" in msg.lower()
    assert "status.roboflow.com" in msg
    assert "Export page" in msg


def test_fail_fast_message_points_at_api_key_for_auth_failures() -> None:
    from app.services.integrations.roboflow_export import _fail_fast_message

    msg = _fail_fast_message([403, 403, 403, 403, 403], ["a.jpg: HTTP 403: nope"] * 5)
    assert "Private API Key" in msg


def test_push_version_fails_fast_after_threshold(real_db_session, monkeypatch) -> None:
    """A run where nothing uploads and the first `_FAIL_FAST_AFTER` images
    all 5xx must abort with an actionable error — not attempt every image.

    Pins `_EXPORT_MAX_WORKERS` to 1 to test the exact, deterministic
    sequential count; `test_push_version_fail_fast_bounded_overshoot_under_concurrency`
    below covers the concurrent case, where this can overshoot by up to
    `max_workers - 1` real attempts."""
    import uuid as _uuid

    from app.services.integrations import roboflow_export as mod

    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(mod, "_EXPORT_MAX_WORKERS", 1)

    def _fake_write_yolo_dataset(db, *, version_id, root):
        for split in ("train", "valid", "test"):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
        for i in range(20):
            (root / "images" / "train" / f"img{i:02d}.jpg").write_bytes(_jpeg_bytes())
            (root / "labels" / "train" / f"img{i:02d}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        data_yaml = root / "data.yaml"
        data_yaml.write_text("names: ['cone']\n", encoding="utf-8")
        return data_yaml

    monkeypatch.setattr(mod, "write_yolo_dataset", _fake_write_yolo_dataset)

    attempted: set[str] = set()

    class _Proj:
        def upload(self, *, image_path, **kwargs):
            attempted.add(image_path)
            raise _upload_error(500)

    class _WS:
        def project(self, slug):
            return _Proj()

    class _RF:
        def workspace(self, ws):
            return _WS()

    monkeypatch.setattr(mod, "get_client", lambda db: (_RF(), {}))

    with pytest.raises(mod.RoboflowExportError) as excinfo:
        mod.push_version_to_roboflow(
            real_db_session,
            version_id=_uuid.uuid4(),
            workspace="ws",
            project_slug="proj",
        )

    assert "quota" in str(excinfo.value).lower()
    # Bailed after the threshold — did NOT try all 20 images.
    assert len(attempted) == mod._FAIL_FAST_AFTER


def test_push_version_fail_fast_checkpoints_the_triggering_failure(real_db_session, monkeypatch) -> None:
    """Regression: the fail-fast `raise` happened before `progress_cb` was
    called for the failure that actually triggered it, so `run_roboflow_export`
    never learned about that last failure — the job row's failed_count/
    processed_items understated how many images were actually attempted.
    `progress_cb` must see every failure, including the one that trips
    fail-fast, before the exception propagates.

    Pins `_EXPORT_MAX_WORKERS` to 1 so the callback sequence is
    deterministic — see the note on `test_push_version_fails_fast_after_threshold`."""
    import uuid as _uuid

    from app.services.integrations import roboflow_export as mod

    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(mod, "_EXPORT_MAX_WORKERS", 1)

    def _fake_write_yolo_dataset(db, *, version_id, root):
        for split in ("train", "valid", "test"):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
        for i in range(20):
            (root / "images" / "train" / f"img{i:02d}.jpg").write_bytes(_jpeg_bytes())
            (root / "labels" / "train" / f"img{i:02d}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        data_yaml = root / "data.yaml"
        data_yaml.write_text("names: ['cone']\n", encoding="utf-8")
        return data_yaml

    monkeypatch.setattr(mod, "write_yolo_dataset", _fake_write_yolo_dataset)

    class _Proj:
        def upload(self, **kwargs):
            raise _upload_error(500)

    class _WS:
        def project(self, slug):
            return _Proj()

    class _RF:
        def workspace(self, ws):
            return _WS()

    monkeypatch.setattr(mod, "get_client", lambda db: (_RF(), {}))

    seen: list[tuple[int, int, int]] = []
    with pytest.raises(mod.RoboflowExportError):
        mod.push_version_to_roboflow(
            real_db_session,
            version_id=_uuid.uuid4(),
            workspace="ws",
            project_slug="proj",
            progress_cb=lambda cur, total, failed: seen.append((cur, total, failed)),
        )

    # The last call must report the 5th failure, not stop at the 4th —
    # otherwise a caller checkpointing on this callback never learns the
    # triggering failure happened at all.
    assert seen[-1] == (0, 20, mod._FAIL_FAST_AFTER)


def test_push_version_progress_counts_successes_not_attempts(real_db_session, monkeypatch) -> None:
    """The progress callback's first arg must be the running count of images
    that actually reached Roboflow — never the loop index — so a half-failing
    push can't show a bar racing ahead of what landed.

    Pins `_EXPORT_MAX_WORKERS` to 1 so the exact interleaving of
    successes/failures below is deterministic instead of depending on
    thread scheduling."""
    import uuid as _uuid

    from app.services.integrations import roboflow_export as mod

    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(mod, "_EXPORT_MAX_WORKERS", 1)

    def _fake_write_yolo_dataset(db, *, version_id, root):
        for split in ("train", "valid", "test"):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
        for i in range(6):
            (root / "images" / "train" / f"img{i}.jpg").write_bytes(_jpeg_bytes())
        (root / "data.yaml").write_text("names: ['cone']\n", encoding="utf-8")
        return root / "data.yaml"

    monkeypatch.setattr(mod, "write_yolo_dataset", _fake_write_yolo_dataset)

    n = {"i": 0}

    class _Proj:
        def upload(self, **kwargs):
            i = n["i"]
            n["i"] += 1
            if i % 2 == 1:  # every other image is rejected (400: no retry, no fail-fast — one already landed)
                raise _upload_error(400)

    class _WS:
        def project(self, slug):
            return _Proj()

    class _RF:
        def workspace(self, ws):
            return _WS()

    monkeypatch.setattr(mod, "get_client", lambda db: (_RF(), {}))

    seen: list[tuple[int, int, int]] = []
    result = mod.push_version_to_roboflow(
        real_db_session,
        version_id=_uuid.uuid4(),
        workspace="ws",
        project_slug="proj",
        progress_cb=lambda cur, total, fail: seen.append((cur, total, fail)),
    )
    uploaded = result.new_images + result.annotations_updated + result.unchanged
    failed = result.failed

    assert (uploaded, failed) == (3, 3)
    assert seen[0] == (0, 6, 0)  # primed once the count is known
    assert [c for c, _, _ in seen] == [0, 1, 1, 2, 2, 3, 3]  # success count, monotonic, never the index
    assert [f for _, _, f in seen] == [0, 0, 1, 1, 2, 2, 3]


def test_push_version_uploads_run_concurrently(real_db_session, monkeypatch) -> None:
    """Regression guard for the whole point of the windowed executor in
    `push_version_to_roboflow`: uploads must actually overlap in time, not
    just be dispatched through a thread pool that happens to run them one
    at a time. Each fake upload blocks briefly while holding a counter of
    how many are concurrently inside `.upload()` — if that peak never rises
    above 1, uploads are still effectively sequential."""
    import threading
    import uuid as _uuid

    from app.services.integrations import roboflow_export as mod

    def _fake_write_yolo_dataset(db, *, version_id, root):
        for split in ("train", "valid", "test"):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
        for i in range(8):
            (root / "images" / "train" / f"img{i}.jpg").write_bytes(_jpeg_bytes())
        (root / "data.yaml").write_text("names: ['cone']\n", encoding="utf-8")
        return root / "data.yaml"

    monkeypatch.setattr(mod, "write_yolo_dataset", _fake_write_yolo_dataset)
    monkeypatch.setattr(mod, "_EXPORT_MAX_WORKERS", 4)

    lock = threading.Lock()
    active = {"current": 0, "peak": 0}

    class _Proj:
        def upload(self, **kwargs):
            with lock:
                active["current"] += 1
                active["peak"] = max(active["peak"], active["current"])
            threading.Event().wait(0.05)
            with lock:
                active["current"] -= 1

    class _WS:
        def project(self, slug):
            return _Proj()

    class _RF:
        def workspace(self, ws):
            return _WS()

    monkeypatch.setattr(mod, "get_client", lambda db: (_RF(), {}))

    result = mod.push_version_to_roboflow(
        real_db_session, version_id=_uuid.uuid4(), workspace="ws", project_slug="proj"
    )
    uploaded = result.new_images + result.annotations_updated + result.unchanged
    failed = result.failed

    assert (uploaded, failed) == (8, 0)
    assert active["peak"] > 1


def test_push_version_fail_fast_bounded_overshoot_under_concurrency(real_db_session, monkeypatch) -> None:
    """With concurrency > 1, fail-fast can't stop at exactly `_FAIL_FAST_AFTER`
    attempts (some are already in flight before the threshold is noticed on
    the main thread) — but the overshoot must stay bounded by
    `_EXPORT_MAX_WORKERS - 1`, not degrade back into "attempt every image."""
    import threading
    import uuid as _uuid

    from app.services.integrations import roboflow_export as mod

    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(mod, "_EXPORT_MAX_WORKERS", 4)

    def _fake_write_yolo_dataset(db, *, version_id, root):
        for split in ("train", "valid", "test"):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
        for i in range(20):
            (root / "images" / "train" / f"img{i:02d}.jpg").write_bytes(_jpeg_bytes())
            (root / "labels" / "train" / f"img{i:02d}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        data_yaml = root / "data.yaml"
        data_yaml.write_text("names: ['cone']\n", encoding="utf-8")
        return data_yaml

    monkeypatch.setattr(mod, "write_yolo_dataset", _fake_write_yolo_dataset)

    # A set, not a list: each image is retried up to `_UPLOAD_MAX_ATTEMPTS`
    # times on a 500 before counting as one failed image — this tracks
    # distinct images attempted, matching what `_FAIL_FAST_AFTER` counts.
    attempted: set[str] = set()
    lock = threading.Lock()

    class _Proj:
        def upload(self, *, image_path, **kwargs):
            with lock:
                attempted.add(image_path)
            raise _upload_error(500)

    class _WS:
        def project(self, slug):
            return _Proj()

    class _RF:
        def workspace(self, ws):
            return _WS()

    monkeypatch.setattr(mod, "get_client", lambda db: (_RF(), {}))

    with pytest.raises(mod.RoboflowExportError) as excinfo:
        mod.push_version_to_roboflow(
            real_db_session, version_id=_uuid.uuid4(), workspace="ws", project_slug="proj"
        )

    assert "quota" in str(excinfo.value).lower()
    assert mod._FAIL_FAST_AFTER <= len(attempted) <= mod._FAIL_FAST_AFTER + mod._EXPORT_MAX_WORKERS - 1
    assert len(attempted) < 20


def test_push_version_all_duplicates_updates_annotations_not_skipped(real_db_session, monkeypatch) -> None:
    """Roboflow reports every upload as a duplicate (byte-identical image
    already in the project — this app's own normal
    import-then-annotate-then-push workflow). That must NOT read as
    nothing happened: each duplicate's annotation is still written (via
    the SDK's own image-id-from-duplicate-response + save_annotation, now
    with overwrite=True so it isn't silently rejected), counted as
    `annotations_updated`, and the note says so plainly — no batch lookup,
    since nothing new landed to file a review job for."""
    import uuid as _uuid

    from app.services.integrations import roboflow_export as mod

    def _fake_write_yolo_dataset(db, *, version_id, root):
        for split in ("train", "valid", "test"):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
        for i in range(3):
            (root / "images" / "train" / f"img{i}.jpg").write_bytes(_jpeg_bytes())
            (root / "labels" / "train" / f"img{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (root / "data.yaml").write_text("names: ['cone']\n", encoding="utf-8")
        return root / "data.yaml"

    monkeypatch.setattr(mod, "write_yolo_dataset", _fake_write_yolo_dataset)

    get_batches_calls = []
    upload_calls = []

    class _Proj:
        def upload(self, **kwargs):
            upload_calls.append(kwargs)
            return [{"image": {"id": "existing-image-id", "success": False, "duplicate": True}}]

        def get_batches(self):
            get_batches_calls.append(1)
            return {"batches": []}

    class _WS:
        def project(self, slug):
            return _Proj()

    class _RF:
        def workspace(self, ws):
            return _WS()

    monkeypatch.setattr(mod, "get_client", lambda db: (_RF(), {"default_labeler_email": "a@b.com"}))

    result = mod.push_version_to_roboflow(
        real_db_session, version_id=_uuid.uuid4(), workspace="ws", project_slug="proj"
    )

    assert (result.new_images, result.annotations_updated, result.unchanged, result.failed) == (0, 3, 0, 0)
    assert not get_batches_calls  # never searched for a batch that was never created
    assert result.note is not None
    assert "3 existing image" in result.note
    assert "updated" in result.note
    assert "couldn't find batch" not in result.note.lower()
    # the actual bug this whole plan exists to fix: overwrite must be requested
    assert all(kwargs.get("annotation_overwrite") is True for kwargs in upload_calls)


def test_push_version_mixed_new_and_duplicate_still_files_review_job_and_reports_both(
    real_db_session, monkeypatch
) -> None:
    """At least one genuinely new upload means the batch really was created
    on Roboflow — the review job must still be filed for it normally, and
    the note must mention both the new upload and the updated duplicate,
    not stay silent just because the review job itself succeeded."""
    import uuid as _uuid

    from app.services.integrations import roboflow_export as mod

    def _fake_write_yolo_dataset(db, *, version_id, root):
        for split in ("train", "valid", "test"):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
        for i in range(2):
            (root / "images" / "train" / f"img{i}.jpg").write_bytes(_jpeg_bytes())
            (root / "labels" / "train" / f"img{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (root / "data.yaml").write_text("names: ['cone']\n", encoding="utf-8")
        return root / "data.yaml"

    monkeypatch.setattr(mod, "write_yolo_dataset", _fake_write_yolo_dataset)

    calls = {"n": 0}

    class _Proj:
        def upload(self, *, batch_name, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return [{"image": {"id": "new-image-id", "success": True, "duplicate": False}}]
            return [{"image": {"id": "existing-image-id", "success": False, "duplicate": True}}]

        def get_batches(self):
            return {"batches": [{"id": "batch-id-1", "name": "autolabelflow", "images": 1}]}

        def create_annotation_job(self, **kwargs):
            return {"id": "job-1"}

    class _WS:
        def project(self, slug):
            return _Proj()

    class _RF:
        def workspace(self, ws):
            return _WS()

    monkeypatch.setattr(mod, "get_client", lambda db: (_RF(), {"default_labeler_email": "a@b.com"}))

    result = mod.push_version_to_roboflow(
        real_db_session,
        version_id=_uuid.uuid4(),
        workspace="ws",
        project_slug="proj",
        custom_batch_name="autolabelflow",
    )

    assert (result.new_images, result.annotations_updated, result.failed) == (1, 1, 0)
    assert result.note is not None
    assert "1 new image" in result.note
    assert "1 existing image" in result.note


def test_push_version_known_roboflow_id_routes_through_save_annotation_not_upload(
    real_db_session, monkeypatch, unique_name: str
) -> None:
    """Core of Task 5: an Image row with a roboflow_image_id matching the
    export target's workspace/project must skip Project.upload() entirely
    and call Project.save_annotation() directly instead."""
    import uuid as _uuid

    from app.models.dataset import Dataset
    from app.models.image import Image, ImageSourceType
    from app.models.project import Project
    from app.services.integrations import roboflow_export as mod

    project = Project(name=unique_name, slug=unique_name, class_config=[{"id": 0, "name": "cone"}])
    real_db_session.add(project)
    real_db_session.flush()
    dataset = Dataset(project_id=project.id, name="ds")
    real_db_session.add(dataset)
    real_db_session.flush()
    image = Image(
        project_id=project.id,
        dataset_id=dataset.id,
        storage_key="k",
        original_filename="a.jpg",
        width=64,
        height=48,
        source_type=ImageSourceType.UPLOAD,
        roboflow_image_id="known-rf-image-id",
        roboflow_workspace="ws",
        roboflow_project_slug="proj",
    )
    real_db_session.add(image)
    real_db_session.commit()
    real_db_session.refresh(image)

    def _fake_write_yolo_dataset(db, *, version_id, root):
        (root / "images" / "train").mkdir(parents=True)
        (root / "labels" / "train").mkdir(parents=True)
        (root / "images" / "valid").mkdir(parents=True)
        (root / "labels" / "valid").mkdir(parents=True)
        (root / "images" / "test").mkdir(parents=True)
        (root / "labels" / "test").mkdir(parents=True)
        (root / "images" / "train" / f"{image.id}.jpg").write_bytes(_jpeg_bytes())
        (root / "labels" / "train" / f"{image.id}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (root / "data.yaml").write_text("names: ['cone']\n", encoding="utf-8")
        return root / "data.yaml"

    monkeypatch.setattr(mod, "write_yolo_dataset", _fake_write_yolo_dataset)

    upload_calls = []
    save_calls = []

    class _Proj:
        def upload(self, **kwargs):
            upload_calls.append(kwargs)
            return [{"image": {"id": "should-not-be-called", "success": True, "duplicate": False}}]

        def save_annotation(self, **kwargs):
            save_calls.append(kwargs)
            return {"success": True}, 0.0, 0

        def get_batches(self):
            return {"batches": []}

    class _WS:
        def project(self, slug):
            return _Proj()

    class _RF:
        def workspace(self, ws):
            return _WS()

    monkeypatch.setattr(mod, "get_client", lambda db: (_RF(), {"default_labeler_email": "a@b.com"}))

    result = mod.push_version_to_roboflow(
        real_db_session, version_id=_uuid.uuid4(), workspace="ws", project_slug="proj"
    )

    assert not upload_calls
    assert len(save_calls) == 1
    assert save_calls[0]["image_id"] == "known-rf-image-id"
    assert save_calls[0]["annotation_overwrite"] is True
    assert save_calls[0]["annotation_labelmap"] == {0: "cone"}  # loaded, not the bare yaml path
    assert (result.new_images, result.annotations_updated, result.unchanged, result.failed) == (0, 1, 0, 0)


def test_push_version_known_roboflow_id_warn_response_counts_as_unchanged(
    real_db_session, monkeypatch, unique_name: str
) -> None:
    """Regression: if Roboflow's 409 "already annotated" response ever slips
    past `annotation_overwrite=True` (a behavior change on their end, a
    future SDK change, anything), `rfapi.save_annotation` returns
    `{"warn": "already annotated"}` instead of raising — nothing actually
    wrote on Roboflow's side. `_save_annotation_only` must not report that
    as `ANNOTATION_UPDATED` (the exact bug this branch exists to fix,
    relocated); it must be counted as `unchanged` instead."""
    import uuid as _uuid

    from app.models.dataset import Dataset
    from app.models.image import Image, ImageSourceType
    from app.models.project import Project
    from app.services.integrations import roboflow_export as mod

    project = Project(name=unique_name, slug=unique_name, class_config=[{"id": 0, "name": "cone"}])
    real_db_session.add(project)
    real_db_session.flush()
    dataset = Dataset(project_id=project.id, name="ds")
    real_db_session.add(dataset)
    real_db_session.flush()
    image = Image(
        project_id=project.id,
        dataset_id=dataset.id,
        storage_key="k",
        original_filename="a.jpg",
        width=64,
        height=48,
        source_type=ImageSourceType.UPLOAD,
        roboflow_image_id="known-rf-image-id",
        roboflow_workspace="ws",
        roboflow_project_slug="proj",
    )
    real_db_session.add(image)
    real_db_session.commit()
    real_db_session.refresh(image)

    def _fake_write_yolo_dataset(db, *, version_id, root):
        (root / "images" / "train").mkdir(parents=True)
        (root / "labels" / "train").mkdir(parents=True)
        (root / "images" / "valid").mkdir(parents=True)
        (root / "labels" / "valid").mkdir(parents=True)
        (root / "images" / "test").mkdir(parents=True)
        (root / "labels" / "test").mkdir(parents=True)
        (root / "images" / "train" / f"{image.id}.jpg").write_bytes(_jpeg_bytes())
        (root / "labels" / "train" / f"{image.id}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (root / "data.yaml").write_text("names: ['cone']\n", encoding="utf-8")
        return root / "data.yaml"

    monkeypatch.setattr(mod, "write_yolo_dataset", _fake_write_yolo_dataset)

    upload_calls = []
    save_calls = []

    class _Proj:
        def upload(self, **kwargs):
            upload_calls.append(kwargs)
            return [{"image": {"id": "should-not-be-called", "success": True, "duplicate": False}}]

        def save_annotation(self, **kwargs):
            save_calls.append(kwargs)
            return {"warn": "already annotated"}, 0.0, 0

        def get_batches(self):
            return {"batches": []}

    class _WS:
        def project(self, slug):
            return _Proj()

    class _RF:
        def workspace(self, ws):
            return _WS()

    monkeypatch.setattr(mod, "get_client", lambda db: (_RF(), {"default_labeler_email": "a@b.com"}))

    result = mod.push_version_to_roboflow(
        real_db_session, version_id=_uuid.uuid4(), workspace="ws", project_slug="proj"
    )

    assert not upload_calls
    assert len(save_calls) == 1
    assert (result.new_images, result.annotations_updated, result.unchanged, result.failed) == (0, 0, 1, 0)


def test_push_version_known_roboflow_id_different_project_falls_back_to_upload(
    real_db_session, monkeypatch, unique_name: str
) -> None:
    """A stored id from a *different* Roboflow project than the one being
    exported to must not be trusted — falls back to a normal upload."""
    import uuid as _uuid

    from app.models.dataset import Dataset
    from app.models.image import Image, ImageSourceType
    from app.models.project import Project
    from app.services.integrations import roboflow_export as mod

    project = Project(name=unique_name, slug=unique_name, class_config=[{"id": 0, "name": "cone"}])
    real_db_session.add(project)
    real_db_session.flush()
    dataset = Dataset(project_id=project.id, name="ds")
    real_db_session.add(dataset)
    real_db_session.flush()
    image = Image(
        project_id=project.id,
        dataset_id=dataset.id,
        storage_key="k",
        original_filename="a.jpg",
        width=64,
        height=48,
        source_type=ImageSourceType.UPLOAD,
        roboflow_image_id="known-rf-image-id",
        roboflow_workspace="ws",
        roboflow_project_slug="a-different-project",  # <- mismatch
    )
    real_db_session.add(image)
    real_db_session.commit()
    real_db_session.refresh(image)

    def _fake_write_yolo_dataset(db, *, version_id, root):
        (root / "images" / "train").mkdir(parents=True)
        (root / "labels" / "train").mkdir(parents=True)
        (root / "images" / "valid").mkdir(parents=True)
        (root / "labels" / "valid").mkdir(parents=True)
        (root / "images" / "test").mkdir(parents=True)
        (root / "labels" / "test").mkdir(parents=True)
        (root / "images" / "train" / f"{image.id}.jpg").write_bytes(_jpeg_bytes())
        (root / "labels" / "train" / f"{image.id}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (root / "data.yaml").write_text("names: ['cone']\n", encoding="utf-8")
        return root / "data.yaml"

    monkeypatch.setattr(mod, "write_yolo_dataset", _fake_write_yolo_dataset)

    upload_calls = []
    save_calls = []

    class _Proj:
        def upload(self, **kwargs):
            upload_calls.append(kwargs)
            return [{"image": {"id": "new-id", "success": True, "duplicate": False}}]

        def save_annotation(self, **kwargs):
            save_calls.append(kwargs)
            return {"success": True}

        def get_batches(self):
            return {"batches": []}

    class _WS:
        def project(self, slug):
            return _Proj()

    class _RF:
        def workspace(self, ws):
            return _WS()

    monkeypatch.setattr(mod, "get_client", lambda db: (_RF(), {"default_labeler_email": "a@b.com"}))

    result = mod.push_version_to_roboflow(
        real_db_session, version_id=_uuid.uuid4(), workspace="ws", project_slug="proj"
    )

    assert not save_calls
    assert len(upload_calls) == 1
    assert result.new_images == 1


def test_push_version_stale_known_roboflow_id_404_falls_back_to_upload(
    real_db_session, monkeypatch, unique_name: str
) -> None:
    """The stored id no longer resolves on Roboflow (image deleted there
    since import) — save_annotation 404s, and that one image falls back to
    a normal upload instead of failing the whole export."""
    import uuid as _uuid

    from app.models.dataset import Dataset
    from app.models.image import Image, ImageSourceType
    from app.models.project import Project
    from app.services.integrations import roboflow_export as mod

    project = Project(name=unique_name, slug=unique_name, class_config=[{"id": 0, "name": "cone"}])
    real_db_session.add(project)
    real_db_session.flush()
    dataset = Dataset(project_id=project.id, name="ds")
    real_db_session.add(dataset)
    real_db_session.flush()
    image = Image(
        project_id=project.id,
        dataset_id=dataset.id,
        storage_key="k",
        original_filename="a.jpg",
        width=64,
        height=48,
        source_type=ImageSourceType.UPLOAD,
        roboflow_image_id="missing-on-roboflow",
        roboflow_workspace="ws",
        roboflow_project_slug="proj",
    )
    real_db_session.add(image)
    real_db_session.commit()
    real_db_session.refresh(image)

    def _fake_write_yolo_dataset(db, *, version_id, root):
        (root / "images" / "train").mkdir(parents=True)
        (root / "labels" / "train").mkdir(parents=True)
        (root / "images" / "valid").mkdir(parents=True)
        (root / "labels" / "valid").mkdir(parents=True)
        (root / "images" / "test").mkdir(parents=True)
        (root / "labels" / "test").mkdir(parents=True)
        (root / "images" / "train" / f"{image.id}.jpg").write_bytes(_jpeg_bytes())
        (root / "labels" / "train" / f"{image.id}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (root / "data.yaml").write_text("names: ['cone']\n", encoding="utf-8")
        return root / "data.yaml"

    monkeypatch.setattr(mod, "write_yolo_dataset", _fake_write_yolo_dataset)

    upload_calls = []

    class _Proj:
        def upload(self, **kwargs):
            upload_calls.append(kwargs)
            return [{"image": {"id": "new-id", "success": True, "duplicate": False}}]

        def save_annotation(self, **kwargs):
            from roboflow.adapters.rfapi import AnnotationSaveError

            raise AnnotationSaveError("not found", status_code=404)

        def get_batches(self):
            return {"batches": []}

    class _WS:
        def project(self, slug):
            return _Proj()

    class _RF:
        def workspace(self, ws):
            return _WS()

    monkeypatch.setattr(mod, "get_client", lambda db: (_RF(), {"default_labeler_email": "a@b.com"}))

    result = mod.push_version_to_roboflow(
        real_db_session, version_id=_uuid.uuid4(), workspace="ws", project_slug="proj"
    )

    assert len(upload_calls) == 1
    assert result.new_images == 1
    assert result.failed == 0


def test_roboflow_export_all_uploads_fail_marks_job_failed(
    connected_roboflow: TestClient, approved_version: tuple[str, str], monkeypatch
) -> None:
    """A version too small to trip fail-fast, but with every image failing,
    must still land as FAILED with a real message — not COMPLETED / 0
    uploaded."""
    _, version_id = approved_version

    def _always_500(self, **kwargs):
        raise _upload_error(500)

    monkeypatch.setattr(_FakeRoboflowProject, "upload", _always_500)

    resp = connected_roboflow.post(
        f"/api/v1/versions/{version_id}/export/roboflow",
        json={"workspace": "my-workspace", "project": "cones"},
    )
    assert resp.status_code == 202, resp.text
    job = resp.json()
    assert job["status"] == "FAILED"
    assert job["uploaded_count"] == 0
    assert job["failed_count"] == 1
    # Honest progress: the one image that failed must NOT count as processed.
    assert job["processed_items"] == 0
    assert "roboflow" in job["error"].lower()
    assert "HTTP 500" in job["error"]

    progress = get_progress(job["id"])
    assert progress is not None
    assert progress.status == "FAILED"
    assert progress.current == 0
    assert progress.failed == 1


def test_roboflow_export_omits_annotation_path_for_unannotated_image(
    connected_roboflow: TestClient, unique_name: str, monkeypatch
) -> None:
    """Regression: `write_yolo_dataset` writes labels/*.txt for every image,
    including an approved image with zero annotations (a legitimate
    "background" image — versioning only filters on APPROVED, not on
    having annotations) — that file exists but is empty, not absent. Every
    such image failed to push with Roboflow's own `HTTP 400: Unrecognized
    annotation format` (confirmed 1:1 against every zero-annotation image
    in a real 73-image push) because `annotation_path` was set to that
    empty file instead of None."""
    project = connected_roboflow.post("/api/v1/projects", json={"name": unique_name}).json()
    connected_roboflow.patch(
        f"/api/v1/projects/{project['id']}", json={"class_config": [{"id": 0, "name": "cone"}]}
    )
    dataset = connected_roboflow.post(f"/api/v1/projects/{project['id']}/datasets", json={"name": "d"}).json()
    image = connected_roboflow.post(
        f"/api/v1/datasets/{dataset['id']}/images", files={"file": ("f.jpg", _jpeg_bytes(), "image/jpeg")}
    ).json()
    connected_roboflow.post(f"/api/v1/images/{image['id']}/approve")  # approved, no annotation added
    version = connected_roboflow.post(f"/api/v1/datasets/{dataset['id']}/versions", json={}).json()

    captured: list[dict] = []
    original_upload = _FakeRoboflowProject.upload

    def _spy_upload(self, **kwargs):
        captured.append(kwargs)
        return original_upload(self, **kwargs)

    monkeypatch.setattr(_FakeRoboflowProject, "upload", _spy_upload)

    resp = connected_roboflow.post(
        f"/api/v1/versions/{version['id']}/export/roboflow",
        json={"workspace": "my-workspace", "project": "cones"},
    )
    assert resp.status_code == 202, resp.text
    assert resp.json()["status"] == "COMPLETED"
    assert resp.json()["uploaded_count"] == 1

    assert captured
    assert captured[0]["annotation_path"] is None


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


def test_roboflow_job_model_has_new_images_and_annotations_updated_counts(
    real_db_session, unique_name: str
) -> None:
    from app.models.roboflow_job import RoboflowJob, RoboflowJobKind
    from app.models.project import Project

    # `project_id` is a real FK (Postgres enforces it in this test env, same
    # as the `RoboflowJob(...)` constructions already in this file around
    # lines 826 and 861 — both use a real project id from a created
    # project, never a bare random uuid), so a real Project row comes first.
    project = Project(name=unique_name, slug=unique_name, class_config=[{"id": 0, "name": "cone"}])
    real_db_session.add(project)
    real_db_session.commit()

    job = RoboflowJob(
        project_id=project.id,
        kind=RoboflowJobKind.EXPORT,
        workspace="ws",
        project_slug="proj",
    )
    real_db_session.add(job)
    real_db_session.commit()
    real_db_session.refresh(job)

    assert job.new_images_count == 0
    assert job.annotations_updated_count == 0

    job.new_images_count = 3
    job.annotations_updated_count = 7
    real_db_session.commit()
    real_db_session.refresh(job)
    assert (job.new_images_count, job.annotations_updated_count) == (3, 7)
