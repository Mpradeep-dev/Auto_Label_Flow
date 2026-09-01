"""Import labelled images from an Azure Blob prefix *by reference* — no
byte copy. The Azure `ContainerClient` is faked at the module boundary
(house convention: fake the external boundary, keep the suite offline);
the service and the background task are exercised for real against a
committing session, same pattern as `test_roboflow_jobs.py`.
"""
from __future__ import annotations

import io
import json
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage

import app.services.integrations.azure_blob_import as blob_import_mod
from app.services.integrations.azure_blob_import import AzureBlobImportError, import_azure_blob_prefix


def _jpeg(w: int = 64, h: int = 48) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (w, h), color=(70, 90, 110)).save(buf, format="JPEG")
    return buf.getvalue()


_YOLO_BLOBS: dict[str, bytes] = {
    "prod-yolo/data.yaml": b"names: ['ball', 'cone']\n",
    "prod-yolo/train/images/img1.jpg": _jpeg(),
    "prod-yolo/train/labels/img1.txt": b"0 0.5 0.5 0.2 0.2\n1 0.25 0.25 0.1 0.1\n",
    "prod-yolo/valid/images/img2.jpg": _jpeg(),
    "prod-yolo/valid/labels/img2.txt": b"1 0.5 0.5 0.4 0.4\n",
    # A blob outside the prefix — must be ignored by name_starts_with.
    "other/stray.jpg": _jpeg(),
}

_COCO_DOC = {
    "images": [{"id": 1, "file_name": "pic1.jpg", "width": 64, "height": 48}],
    "annotations": [{"id": 1, "image_id": 1, "category_id": 7, "bbox": [10, 10, 20, 15], "segmentation": []}],
    "categories": [{"id": 7, "name": "ball"}],
}
_COCO_BLOBS: dict[str, bytes] = {
    "prod-coco/_annotations.coco.json": json.dumps(_COCO_DOC).encode("utf-8"),
    "prod-coco/images/pic1.jpg": _jpeg(),
}


class _FakeDownloader:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def readall(self) -> bytes:
        return self._data


class _FakeContainer:
    def __init__(self, blobs: dict[str, bytes]) -> None:
        self._blobs = blobs
        self.ranged_reads: list[str] = []

    def list_blobs(self, name_starts_with: str = ""):
        for name in sorted(self._blobs):
            if name.startswith(name_starts_with):
                yield SimpleNamespace(name=name)

    def download_blob(self, name: str, offset: int | None = None, length: int | None = None) -> _FakeDownloader:
        data = self._blobs[name]
        if offset is not None:
            self.ranged_reads.append(name)
            end = offset + length if length is not None else None
            data = data[offset:end]
        return _FakeDownloader(data)


@pytest.fixture()
def fake_yolo(monkeypatch) -> _FakeContainer:
    container = _FakeContainer(_YOLO_BLOBS)
    monkeypatch.setattr(blob_import_mod, "_container_client", lambda: container)
    return container


@pytest.fixture()
def fake_coco(monkeypatch) -> _FakeContainer:
    container = _FakeContainer(_COCO_BLOBS)
    monkeypatch.setattr(blob_import_mod, "_container_client", lambda: container)
    return container


def _project(real_client: TestClient, name: str) -> str:
    return real_client.post("/api/v1/projects", json={"name": name}).json()["id"]


def test_yolo_prefix_imports_by_reference(real_client: TestClient, real_db_session, unique_name, fake_yolo) -> None:
    project_id = _project(real_client, unique_name)

    dataset = import_azure_blob_prefix(
        real_db_session,
        project_id=uuid.UUID(project_id),
        prefix="prod-yolo/",
        label_format="auto",
    )

    from app.models.image import Image, ImageReviewStatus

    images = list(real_db_session.query(Image).filter(Image.dataset_id == dataset.id).order_by(Image.storage_key))
    assert [i.storage_key for i in images] == [
        "prod-yolo/train/images/img1.jpg",
        "prod-yolo/valid/images/img2.jpg",
    ]
    assert all(i.is_external is True for i in images)
    assert all((i.width, i.height) == (64, 48) for i in images)
    assert all(i.review_status == ImageReviewStatus.APPROVED for i in images)

    project = real_client.get(f"/api/v1/projects/{project_id}").json()
    assert {c["name"] for c in project["class_config"]} == {"ball", "cone"}

    anns = real_client.get(f"/api/v1/images/{images[0].id}/annotations").json()
    assert sorted(a["class_name"] for a in anns) == ["ball", "cone"]
    ball = next(a for a in anns if a["class_name"] == "ball")
    assert ball["x1"] == pytest.approx(0.4)
    assert ball["y1"] == pytest.approx(0.4)
    assert ball["x2"] == pytest.approx(0.6)
    assert ball["y2"] == pytest.approx(0.6)
    assert ball["source"] == "HUMAN"


def test_coco_prefix_uses_json_dims_and_resolves_by_name(
    real_client: TestClient, real_db_session, unique_name, fake_coco
) -> None:
    project_id = _project(real_client, unique_name)

    dataset = import_azure_blob_prefix(
        real_db_session,
        project_id=uuid.UUID(project_id),
        prefix="prod-coco/",
        label_format="auto",
    )

    from app.models.image import Image

    image = real_db_session.query(Image).filter(Image.dataset_id == dataset.id).one()
    assert image.storage_key == "prod-coco/images/pic1.jpg"
    assert image.is_external is True
    assert (image.width, image.height) == (64, 48)
    # width/height came from the COCO json — no header read for this blob.
    assert "prod-coco/images/pic1.jpg" not in fake_coco.ranged_reads

    anns = real_client.get(f"/api/v1/images/{image.id}/annotations").json()
    assert len(anns) == 1
    assert anns[0]["class_name"] == "ball"
    assert anns[0]["x1"] == pytest.approx(10 / 64)
    assert anns[0]["x2"] == pytest.approx(30 / 64)


def test_cancel_stops_before_first_image_but_keeps_dataset(
    real_client: TestClient, real_db_session, unique_name, fake_yolo
) -> None:
    project_id = _project(real_client, unique_name)

    dataset = import_azure_blob_prefix(
        real_db_session,
        project_id=uuid.UUID(project_id),
        prefix="prod-yolo/",
        label_format="yolo",
        should_cancel=lambda: True,
    )

    from app.models.image import Image

    assert real_db_session.query(Image).filter(Image.dataset_id == dataset.id).count() == 0
    assert dataset.id is not None


def test_auto_detect_raises_when_no_labels(real_client: TestClient, real_db_session, unique_name, monkeypatch) -> None:
    monkeypatch.setattr(
        blob_import_mod, "_container_client", lambda: _FakeContainer({"bare/img.jpg": _jpeg()})
    )
    project_id = _project(real_client, unique_name)

    with pytest.raises(AzureBlobImportError):
        import_azure_blob_prefix(
            real_db_session, project_id=uuid.UUID(project_id), prefix="bare/", label_format="auto"
        )


def test_endpoint_rejects_when_storage_backend_not_azure(client: TestClient, unique_name) -> None:
    project_id = client.post("/api/v1/projects", json={"name": unique_name}).json()["id"]
    resp = client.post(
        f"/api/v1/projects/{project_id}/import/azure-blob",
        json={"prefix": "prod-yolo/", "label_format": "auto"},
    )
    assert resp.status_code == 400
    assert "azure" in resp.json()["detail"].lower()


def test_job_task_completes(real_client: TestClient, real_db_session, unique_name, monkeypatch) -> None:
    from app.models.blob_import_job import BlobImportJob, BlobImportJobStatus
    from app.workers.tasks.blob_import import run_blob_import

    monkeypatch.setattr(blob_import_mod, "_container_client", lambda: _FakeContainer(_YOLO_BLOBS))
    project_id = _project(real_client, unique_name)

    job = BlobImportJob(project_id=uuid.UUID(project_id), prefix="prod-yolo/", label_format="auto")
    real_db_session.add(job)
    real_db_session.commit()
    real_db_session.refresh(job)

    run_blob_import(str(job.id))

    real_db_session.refresh(job)
    assert job.status == BlobImportJobStatus.COMPLETED
    assert job.total_items == 2
    assert job.processed_items == 2
    assert job.result_dataset_id is not None
    assert job.error is None


def test_job_task_cancel_lands_cancelled(real_client: TestClient, real_db_session, unique_name, monkeypatch) -> None:
    from app.models.blob_import_job import BlobImportJob, BlobImportJobStatus
    from app.workers.progress import request_cancel
    from app.workers.tasks.blob_import import run_blob_import

    monkeypatch.setattr(blob_import_mod, "_container_client", lambda: _FakeContainer(_YOLO_BLOBS))
    project_id = _project(real_client, unique_name)

    job = BlobImportJob(project_id=uuid.UUID(project_id), prefix="prod-yolo/", label_format="yolo")
    real_db_session.add(job)
    real_db_session.commit()
    real_db_session.refresh(job)

    request_cancel(str(job.id))
    run_blob_import(str(job.id))

    real_db_session.refresh(job)
    assert job.status == BlobImportJobStatus.CANCELLED
    assert job.processed_items == 0
    assert job.result_dataset_id is not None


def test_deleting_external_images_leaves_the_blob_untouched(
    real_client: TestClient, real_db_session, unique_name, fake_yolo, monkeypatch
) -> None:
    """`delete_image` / `delete_dataset` must not call storage.delete for an
    is_external row — the blob belongs to whoever put it in the container."""
    import app.api.v1.datasets as datasets_mod
    import app.api.v1.images as images_mod

    deleted_keys: list[str] = []

    class _SpyStorage:
        def delete(self, key: str) -> None:
            deleted_keys.append(key)

    monkeypatch.setattr(images_mod, "get_storage", lambda: _SpyStorage())
    monkeypatch.setattr(datasets_mod, "get_storage", lambda: _SpyStorage())

    project_id = _project(real_client, unique_name)
    dataset = import_azure_blob_prefix(
        real_db_session, project_id=uuid.UUID(project_id), prefix="prod-yolo/", label_format="yolo"
    )

    from app.models.image import Image

    first = real_db_session.query(Image).filter(Image.dataset_id == dataset.id).order_by(Image.storage_key).first()
    assert real_client.delete(f"/api/v1/images/{first.id}").status_code == 204
    assert real_client.delete(f"/api/v1/datasets/{dataset.id}").status_code == 204
    assert deleted_keys == []
