"""COCO detection export — one of the two formats (with CVAT-XML) CVAT's
own UI can import directly, so this is the bridge into CVAT without a live
API integration: export here, then in CVAT use Task > Upload annotations >
format "COCO 1.0" (or create a task from the same zip's images first).

Matches the shape CVAT itself produces when you export a task as "COCO
1.0" — `images/<filename>` (flat, no train/val/test split: neither COCO nor
a CVAT task has that concept) plus `annotations/instances_default.json` —
so a round trip through CVAT (export from here, edit there, export back)
lands on a format this app already knows how to read via
`import_coco_dataset`.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.security import safe_storage_key
from app.models.dataset import Dataset
from app.models.dataset_version import DatasetVersion
from app.services.dataset.version_data import VersionDataError, load_version_data, out_image_filename
from app.services.storage.factory import get_storage


class ExportError(RuntimeError):
    pass


def write_coco_dataset(db: Session, *, version_id: uuid.UUID, root: Path) -> Path:
    """Materializes `images/` + `annotations/instances_default.json` under
    `root`. Returns the json path. Split assignment (train/val/test) isn't
    representable in plain COCO — each image's split is included as a
    custom `annotate_split` field on its `images` entry so it round-trips
    through anything that preserves unknown JSON fields, but is not part of
    the COCO spec itself."""
    try:
        data = load_version_data(db, version_id=version_id)
    except VersionDataError as exc:
        raise ExportError(str(exc)) from exc

    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    storage = get_storage()

    categories = [{"id": class_id, "name": name, "supercategory": "none"} for class_id, name in data.class_names.items()]

    coco_images = []
    coco_annotations = []
    next_annotation_id = 1
    # COCO's `image_id` is a small sequential int, not our UUIDs — keep our
    # own id as `annotate_image_id` alongside it so a re-import can match
    # images back up without guessing from `file_name` alone.
    for i, vi in enumerate(data.images, start=1):
        out_name = out_image_filename(vi.image)
        (images_dir / out_name).write_bytes(storage.read_bytes(vi.image.storage_key))

        coco_images.append(
            {
                "id": i,
                "file_name": out_name,
                "width": vi.image.width,
                "height": vi.image.height,
                "annotate_image_id": str(vi.image.id),
                "annotate_split": vi.split,
            }
        )

        for event in vi.events:
            x = event.x1 * vi.image.width
            y = event.y1 * vi.image.height
            w = (event.x2 - event.x1) * vi.image.width
            h = (event.y2 - event.y1) * vi.image.height
            coco_annotations.append(
                {
                    "id": next_annotation_id,
                    "image_id": i,
                    "category_id": event.class_id,
                    "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                    "area": round(w * h, 2),
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            next_annotation_id += 1

    coco = {
        "info": {"description": f"{data.dataset.name} — exported from Auto Annotation"},
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }

    annotations_dir = root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    json_path = annotations_dir / "instances_default.json"
    json_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    if data.skipped_image_ids:
        (root / "SKIPPED_IMAGES.txt").write_text("\n".join(data.skipped_image_ids), encoding="utf-8")

    return json_path


def export_coco(db: Session, *, version_id: uuid.UUID) -> str:
    """Builds the COCO directory structure, zips it, uploads to storage,
    returns the storage key. Synchronous — same scale tradeoff as
    `export_yolo`."""
    version = db.get(DatasetVersion, version_id)
    if version is None:
        raise ExportError(f"No dataset version with id {version_id}")
    dataset = db.get(Dataset, version.dataset_id)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "dataset"
        root.mkdir()
        write_coco_dataset(db, version_id=version_id, root=root)

        archive_base = str(Path(tmp) / "export")
        archive_path = shutil.make_archive(archive_base, "zip", root_dir=root)

        key = safe_storage_key(
            str(dataset.project_id),
            str(dataset.id),
            "exports",
            original_filename=f"{dataset.name}-v{version.version_number}-coco.zip",
        )
        get_storage().upload(Path(archive_path), key, content_type="application/zip")

    return key
