"""COCO / CVAT-XML export and import — the CVAT round trip: export from
here in a format CVAT's own UI can import, and import back a zip shaped
like what CVAT itself exports. Exercised through the real service functions
and, for the upload boundary, the real API (multipart upload)."""
from __future__ import annotations

import io
import json
import uuid
import xml.etree.ElementTree as ET
import zipfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage
from sqlalchemy.orm import Session

from app.models.annotation import AnnotationSource
from app.models.dataset import Dataset
from app.models.image import Image, ImageReviewStatus
from app.models.project import Project
from app.services.annotation import service as annotation_service
from app.services.dataset.export_coco import export_coco
from app.services.dataset.export_cvat import export_cvat
from app.services.dataset.import_coco import import_coco_zip
from app.services.dataset.import_cvat import import_cvat_zip
from app.services.dataset.versioning import create_version
from app.services.storage.factory import get_storage


@pytest.fixture()
def project(db_session: Session) -> Project:
    p = Project(
        name=f"p-{uuid.uuid4().hex[:8]}",
        slug=f"p-{uuid.uuid4().hex[:8]}",
        class_config=[{"id": 0, "name": "ball"}, {"id": 1, "name": "cone"}],
    )
    db_session.add(p)
    db_session.commit()
    db_session.refresh(p)
    return p


@pytest.fixture()
def dataset(db_session: Session, project: Project) -> Dataset:
    d = Dataset(project_id=project.id, name="d")
    db_session.add(d)
    db_session.commit()
    db_session.refresh(d)
    return d


def _jpeg_bytes(w: int = 100, h: int = 80) -> bytes:
    img = PILImage.new("RGB", (w, h), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_approved_image(db_session: Session, dataset: Dataset) -> Image:
    key = f"{dataset.project_id}/{dataset.id}/images/{uuid.uuid4().hex}.jpg"
    get_storage().upload_bytes(_jpeg_bytes(), key, content_type="image/jpeg")
    image = Image(
        project_id=dataset.project_id,
        dataset_id=dataset.id,
        storage_key=key,
        original_filename="frame.jpg",
        width=100,
        height=80,
        review_status=ImageReviewStatus.APPROVED,
    )
    db_session.add(image)
    db_session.commit()
    db_session.refresh(image)
    return image


@pytest.fixture()
def versioned_image(db_session: Session, dataset: Dataset):
    """One approved image with a known box: x1=10,y1=8,x2=50,y2=48 out of
    100x80 -> normalized 0.1,0.1,0.5,0.6, pinned into a version."""
    image = _make_approved_image(db_session, dataset)
    annotation_service.create_annotation(
        db_session,
        image_id=image.id,
        class_id=1,
        class_name="cone",
        x1=0.1,
        y1=0.1,
        x2=0.5,
        y2=0.6,
        confidence=0.9,
        source=AnnotationSource.AUTO,
    )
    version = create_version(db_session, dataset_id=dataset.id, train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)
    return image, version


def test_export_coco_bbox_is_pixel_space_and_correct(db_session: Session, versioned_image) -> None:
    _, version = versioned_image
    key = export_coco(db_session, version_id=version.id)
    zip_bytes = get_storage().read_bytes(key)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        assert "annotations/instances_default.json" in names
        image_files = [n for n in names if n.startswith("images/") and not n.endswith("/")]
        assert len(image_files) == 1

        coco = json.loads(zf.read("annotations/instances_default.json"))
        assert {c["name"] for c in coco["categories"]} == {"ball", "cone"}
        assert len(coco["images"]) == 1
        assert coco["images"][0]["width"] == 100
        assert coco["images"][0]["height"] == 80

        assert len(coco["annotations"]) == 1
        ann = coco["annotations"][0]
        # normalized 0.1,0.1,0.5,0.6 over 100x80 -> bbox [10, 8, 40, 40]
        x, y, w, h = ann["bbox"]
        assert x == pytest.approx(10, abs=0.1)
        assert y == pytest.approx(8, abs=0.1)
        assert w == pytest.approx(40, abs=0.1)
        assert h == pytest.approx(40, abs=0.1)
        cone_category_id = next(c["id"] for c in coco["categories"] if c["name"] == "cone")
        assert ann["category_id"] == cone_category_id


def test_export_cvat_xml_box_is_pixel_space_and_correct(db_session: Session, versioned_image) -> None:
    _, version = versioned_image
    key = export_cvat(db_session, version_id=version.id)
    zip_bytes = get_storage().read_bytes(key)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        assert "annotations.xml" in names
        image_files = [n for n in names if n.startswith("images/") and not n.endswith("/")]
        assert len(image_files) == 1

        root = ET.fromstring(zf.read("annotations.xml"))
        image_el = root.find("image")
        assert image_el is not None
        assert image_el.get("width") == "100"
        assert image_el.get("height") == "80"

        box_el = image_el.find("box")
        assert box_el is not None
        assert box_el.get("label") == "cone"
        assert float(box_el.get("xtl")) == pytest.approx(10, abs=0.1)
        assert float(box_el.get("ytl")) == pytest.approx(8, abs=0.1)
        assert float(box_el.get("xbr")) == pytest.approx(50, abs=0.1)
        assert float(box_el.get("ybr")) == pytest.approx(48, abs=0.1)


def test_coco_round_trip_via_service_functions(db_session: Session, project: Project, versioned_image, tmp_path) -> None:
    _, version = versioned_image
    key = export_coco(db_session, version_id=version.id)
    zip_path = tmp_path / "export.zip"
    zip_path.write_bytes(get_storage().read_bytes(key))

    new_project = Project(name=f"p2-{uuid.uuid4().hex[:8]}", slug=f"p2-{uuid.uuid4().hex[:8]}", class_config=[])
    db_session.add(new_project)
    db_session.commit()
    db_session.refresh(new_project)

    new_dataset = import_coco_zip(db_session, project_id=new_project.id, zip_path=zip_path, dataset_name="reimported")
    assert new_dataset.name == "reimported"

    images = list(db_session.query(Image).filter(Image.dataset_id == new_dataset.id))
    assert len(images) == 1
    assert images[0].width == 100
    assert images[0].height == 80

    annotations = annotation_service.list_annotations_for_image(db_session, images[0].id)
    assert len(annotations) == 1
    assert annotations[0].class_name == "cone"
    assert annotations[0].x1 == pytest.approx(0.1, abs=0.01)
    assert annotations[0].y1 == pytest.approx(0.1, abs=0.01)
    assert annotations[0].x2 == pytest.approx(0.5, abs=0.01)
    assert annotations[0].y2 == pytest.approx(0.6, abs=0.01)
    # Raw import, not curated ground truth from this app — pending, not auto-approved.
    assert images[0].review_status == ImageReviewStatus.PENDING

    db_session.refresh(new_project)
    assert any(c["name"] == "cone" for c in new_project.class_config)


def test_cvat_xml_round_trip_via_service_functions(db_session: Session, project: Project, versioned_image, tmp_path) -> None:
    _, version = versioned_image
    key = export_cvat(db_session, version_id=version.id)
    zip_path = tmp_path / "export.zip"
    zip_path.write_bytes(get_storage().read_bytes(key))

    new_project = Project(name=f"p3-{uuid.uuid4().hex[:8]}", slug=f"p3-{uuid.uuid4().hex[:8]}", class_config=[])
    db_session.add(new_project)
    db_session.commit()
    db_session.refresh(new_project)

    new_dataset = import_cvat_zip(db_session, project_id=new_project.id, zip_path=zip_path, dataset_name=None)
    assert new_dataset.name == "cvat-import"

    images = list(db_session.query(Image).filter(Image.dataset_id == new_dataset.id))
    assert len(images) == 1
    annotations = annotation_service.list_annotations_for_image(db_session, images[0].id)
    assert len(annotations) == 1
    assert annotations[0].class_name == "cone"
    assert annotations[0].x1 == pytest.approx(0.1, abs=0.01)
    assert annotations[0].x2 == pytest.approx(0.5, abs=0.01)


def test_export_endpoints_return_download_urls(client: TestClient, versioned_image) -> None:
    _, version = versioned_image
    resp = client.post(f"/api/v1/versions/{version.id}/export/coco")
    assert resp.status_code == 200, resp.text
    assert resp.json()["coco_download_url"] is not None

    resp2 = client.post(f"/api/v1/versions/{version.id}/export/cvat")
    assert resp2.status_code == 200, resp2.text
    assert resp2.json()["cvat_download_url"] is not None


def test_import_coco_via_api_multipart_upload(client: TestClient, project: Project, versioned_image, db_session: Session) -> None:
    _, version = versioned_image
    key = export_coco(db_session, version_id=version.id)
    zip_bytes = get_storage().read_bytes(key)

    resp = client.post(
        f"/api/v1/projects/{project.id}/import/coco",
        files={"file": ("export.zip", zip_bytes, "application/zip")},
        data={"dataset_name": "via-api"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["name"] == "via-api"

    stats = client.get(f"/api/v1/datasets/{body['id']}/stats").json()
    assert stats["total_images"] == 1


def test_import_cvat_via_api_multipart_upload(client: TestClient, project: Project, versioned_image, db_session: Session) -> None:
    _, version = versioned_image
    key = export_cvat(db_session, version_id=version.id)
    zip_bytes = get_storage().read_bytes(key)

    resp = client.post(
        f"/api/v1/projects/{project.id}/import/cvat",
        files={"file": ("export.zip", zip_bytes, "application/zip")},
    )
    assert resp.status_code == 201, resp.text
    stats = client.get(f"/api/v1/datasets/{resp.json()['id']}/stats").json()
    assert stats["total_images"] == 1


def test_import_coco_rejects_non_zip(client: TestClient, project: Project) -> None:
    resp = client.post(
        f"/api/v1/projects/{project.id}/import/coco",
        files={"file": ("not-a-zip.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400
