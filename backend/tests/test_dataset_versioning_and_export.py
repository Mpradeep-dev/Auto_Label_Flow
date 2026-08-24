"""Dataset versioning + YOLO export, exercised through the real service
functions against a real Postgres session and real local storage."""
from __future__ import annotations

import io
import uuid
import zipfile

import pytest
import yaml
from PIL import Image as PILImage
from sqlalchemy.orm import Session

from app.models.annotation import AnnotationSource
from app.models.dataset import Dataset
from app.models.image import Image, ImageReviewStatus
from app.models.project import Project
from app.models.video import Video
from app.services.annotation import service as annotation_service
from app.services.dataset.export_yolo import export_yolo
from app.services.dataset.versioning import NoApprovedImagesError, create_version
from app.services.storage.factory import get_storage


@pytest.fixture()
def project(db_session: Session) -> Project:
    p = Project(
        name=f"p-{uuid.uuid4().hex[:8]}",
        slug=f"p-{uuid.uuid4().hex[:8]}",
        class_config=[{"id": 0, "name": "ball"}, {"id": 1, "name": "cone"}, {"id": 2, "name": "cone_1"}],
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


def _upload_jpeg_bytes() -> bytes:
    img = PILImage.new("RGB", (100, 80), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_approved_image(db_session: Session, dataset: Dataset, video: Video | None = None) -> Image:
    key = f"{dataset.project_id}/{dataset.id}/images/{uuid.uuid4().hex}.jpg"
    get_storage().upload_bytes(_upload_jpeg_bytes(), key, content_type="image/jpeg")
    image = Image(
        project_id=dataset.project_id,
        dataset_id=dataset.id,
        storage_key=key,
        original_filename="frame.jpg",
        width=100,
        height=80,
        review_status=ImageReviewStatus.APPROVED,
        video_id=video.id if video else None,
    )
    db_session.add(image)
    db_session.commit()
    db_session.refresh(image)
    return image


def test_create_version_requires_approved_images(db_session: Session, dataset: Dataset) -> None:
    with pytest.raises(NoApprovedImagesError):
        create_version(db_session, dataset_id=dataset.id)


def test_version_number_increments(db_session: Session, dataset: Dataset) -> None:
    _make_approved_image(db_session, dataset)
    v1 = create_version(db_session, dataset_id=dataset.id)
    _make_approved_image(db_session, dataset)
    v2 = create_version(db_session, dataset_id=dataset.id)
    assert v1.version_number == 1
    assert v2.version_number == 2


def test_pending_images_are_excluded_from_version(db_session: Session, dataset: Dataset) -> None:
    _make_approved_image(db_session, dataset)
    pending = Image(
        project_id=dataset.project_id,
        dataset_id=dataset.id,
        storage_key="x",
        original_filename="x.jpg",
        width=10,
        height=10,
        review_status=ImageReviewStatus.PENDING,
    )
    db_session.add(pending)
    db_session.commit()

    version = create_version(db_session, dataset_id=dataset.id)
    assert version.total_images == 1


def test_export_writes_correct_yolo_structure_and_normalized_labels(
    db_session: Session, dataset: Dataset, project: Project
) -> None:
    image = _make_approved_image(db_session, dataset)
    # A known box: x1=10,y1=8,x2=50,y2=48 out of 100x80 -> normalized 0.1,0.1,0.5,0.6
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
    key = export_yolo(db_session, version_id=version.id)

    zip_bytes = get_storage().read_bytes(key)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        assert "data.yaml" in names
        label_files = [n for n in names if n.startswith("labels/train/") and n.endswith(".txt")]
        assert len(label_files) == 1
        image_files = [n for n in names if n.startswith("images/train/") and not n.endswith("/")]
        assert len(image_files) == 1

        data_yaml = yaml.safe_load(zf.read("data.yaml"))
        assert data_yaml["names"] == ["ball", "cone", "cone_1"]
        assert data_yaml["nc"] == 3

        label_line = zf.read(label_files[0]).decode().strip()
        class_id, cx, cy, w, h = label_line.split()
        assert class_id == "1"
        assert float(cx) == pytest.approx(0.3, abs=1e-4)  # (0.1+0.5)/2
        assert float(cy) == pytest.approx(0.35, abs=1e-4)  # (0.1+0.6)/2
        assert float(w) == pytest.approx(0.4, abs=1e-4)
        assert float(h) == pytest.approx(0.5, abs=1e-4)


def test_export_is_reproducible_after_later_edits(db_session: Session, dataset: Dataset) -> None:
    """The core versioning guarantee: editing an annotation AFTER a version
    is created must not change that version's export — pins freeze the
    event, not a live reference."""
    image = _make_approved_image(db_session, dataset)
    ann = annotation_service.create_annotation(
        db_session,
        image_id=image.id,
        class_id=1,
        class_name="cone",
        x1=0.1,
        y1=0.1,
        x2=0.2,
        y2=0.2,
        confidence=0.9,
        source=AnnotationSource.AUTO,
    )

    version = create_version(db_session, dataset_id=dataset.id, train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)

    # Now correct the annotation — this must NOT affect the already-created version.
    annotation_service.update_annotation(db_session, annotation_id=ann.id, x1=0.9, y1=0.9, x2=0.95, y2=0.95)

    key = export_yolo(db_session, version_id=version.id)
    zip_bytes = get_storage().read_bytes(key)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        label_files = [n for n in zf.namelist() if n.startswith("labels/") and n.endswith(".txt")]
        line = zf.read(label_files[0]).decode().strip()
        _, cx, cy, _, _ = line.split()
        # Original box centre was (0.15, 0.15) — NOT the corrected (0.925, 0.925).
        assert float(cx) == pytest.approx(0.15, abs=1e-4)
        assert float(cy) == pytest.approx(0.15, abs=1e-4)


def test_export_no_video_spans_two_splits(db_session: Session, dataset: Dataset) -> None:
    videos = [Video(project_id=dataset.project_id, dataset_id=dataset.id, storage_key=f"v{i}", original_filename=f"v{i}.mp4") for i in range(6)]
    for v in videos:
        db_session.add(v)
    db_session.commit()

    image_to_video: dict[str, str] = {}
    for v in videos:
        for _ in range(8):
            img = _make_approved_image(db_session, dataset, video=v)
            image_to_video[str(img.id)] = str(v.id)

    version = create_version(db_session, dataset_id=dataset.id, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=5)

    from app.models.dataset_version import DatasetVersionImage
    from sqlalchemy import select

    rows = list(
        db_session.scalars(select(DatasetVersionImage).where(DatasetVersionImage.dataset_version_id == version.id))
    )
    splits_by_video: dict[str, set[str]] = {}
    for row in rows:
        video_id = image_to_video[str(row.image_id)]
        splits_by_video.setdefault(video_id, set()).add(row.split.value)

    for video_id, splits in splits_by_video.items():
        assert len(splits) == 1, f"video {video_id} spans splits {splits}"
    assert not version.used_frame_level_fallback


def test_export_missing_class_config_errors(db_session: Session) -> None:
    project = Project(name=f"p2-{uuid.uuid4().hex[:8]}", slug=f"p2-{uuid.uuid4().hex[:8]}", class_config=[])
    db_session.add(project)
    db_session.commit()
    dataset = Dataset(project_id=project.id, name="d2")
    db_session.add(dataset)
    db_session.commit()
    db_session.refresh(dataset)

    _make_approved_image(db_session, dataset)
    version = create_version(db_session, dataset_id=dataset.id)

    from app.services.dataset.export_yolo import ExportError

    with pytest.raises(ExportError, match="class_config"):
        export_yolo(db_session, version_id=version.id)
