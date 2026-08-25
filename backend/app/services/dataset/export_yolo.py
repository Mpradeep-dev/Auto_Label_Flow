"""YOLO dataset export (PLAN spec section 12). Reads the PINNED annotation
state via `AnnotationEvent`, never the live `annotations` table — an export
of version N must reproduce byte-identical labels no matter how much
review work has happened on the dataset since.

`write_yolo_dataset` is the shared core: it materializes the directory
structure to any destination path. `export_yolo` wraps it for the
API-facing zip-and-upload flow; `services/training/local_provider.py`
calls it directly to get a plain on-disk folder for `YOLO(...).train()` —
same labels, same guarantee, no zip round-trip needed for training.
"""
from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

import yaml
from sqlalchemy.orm import Session

from app.core.security import safe_storage_key
from app.models.annotation import AnnotationEvent
from app.models.dataset import Dataset
from app.models.dataset_version import DatasetVersion
from app.services.dataset.version_data import VersionDataError, load_version_data, out_image_filename
from app.services.storage.factory import get_storage


class ExportError(RuntimeError):
    pass


def _yolo_line(event: AnnotationEvent) -> str:
    """`class_id center_x center_y width height`, normalized [0,1] — the
    annotation is already stored normalized, so this is pure arithmetic,
    no image dimensions needed."""
    cx = (event.x1 + event.x2) / 2
    cy = (event.y1 + event.y2) / 2
    w = event.x2 - event.x1
    h = event.y2 - event.y1
    return f"{event.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def write_yolo_dataset(db: Session, *, version_id: uuid.UUID, root: Path) -> Path:
    """Materializes images/{train,val,test}, labels/{train,val,test}, and
    data.yaml under `root`. Returns the data.yaml path. Raises ExportError
    if the version or its project's class_config is missing."""
    try:
        data = load_version_data(db, version_id=version_id)
    except VersionDataError as exc:
        raise ExportError(str(exc)) from exc

    storage = get_storage()

    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for vi in data.images:
        out_name = out_image_filename(vi.image)
        dest_image_path = root / "images" / vi.split / out_name
        dest_image_path.write_bytes(storage.read_bytes(vi.image.storage_key))

        label_lines = [_yolo_line(e) for e in vi.events]
        (root / "labels" / vi.split / f"{vi.image.id}.txt").write_text("\n".join(label_lines), encoding="utf-8")

    data_yaml = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(data.class_names),
        "names": [data.class_names[i] for i in sorted(data.class_names)],
    }
    data_yaml_path = root / "data.yaml"
    data_yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    if data.skipped_image_ids:
        (root / "SKIPPED_IMAGES.txt").write_text("\n".join(data.skipped_image_ids), encoding="utf-8")

    return data_yaml_path


def export_yolo(db: Session, *, version_id: uuid.UUID) -> str:
    """Builds the YOLO directory structure, zips it, uploads the zip to
    ObjectStorage, and returns its storage key. Runs synchronously in the
    request — fine at this dataset scale; a large-dataset export would move
    this to a Celery task (same pattern as inference jobs) without changing
    the logic here."""
    version = db.get(DatasetVersion, version_id)
    if version is None:
        raise ExportError(f"No dataset version with id {version_id}")
    dataset = db.get(Dataset, version.dataset_id)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "dataset"
        root.mkdir()
        write_yolo_dataset(db, version_id=version_id, root=root)

        archive_base = str(Path(tmp) / "export")
        archive_path = shutil.make_archive(archive_base, "zip", root_dir=root)

        key = safe_storage_key(
            str(dataset.project_id),
            str(dataset.id),
            "exports",
            original_filename=f"{dataset.name}-v{version.version_number}.zip",
        )
        get_storage().upload(Path(archive_path), key, content_type="application/zip")

    return key
