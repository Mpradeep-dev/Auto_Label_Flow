"""Import a COCO-format zip — the counterpart to `export_coco.py`, and
what makes CVAT a real round trip without a live API integration: export a
CVAT task as "COCO 1.0", hand the zip here, get a new Dataset with the same
images and boxes. Also accepts any other COCO-shaped zip (Roboflow, other
tools) with the same `images/` + `annotations/*.json` layout.

Class identity is resolved by NAME against the project's existing
`class_config`, same rule as the Roboflow importer (PLAN "class taxonomy is
read from the model, never hardcoded") — a COCO `category_id` is only
meaningful within that one export, never assumed to match this project's
own ids.
"""
from __future__ import annotations

import json
import tempfile
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from sqlalchemy.orm import Session

from app.core.security import safe_storage_key
from app.models.annotation import AnnotationSource
from app.models.dataset import Dataset
from app.models.image import Image, ImageSourceType
from app.models.project import Project
from app.services.annotation.service import create_annotation
from app.services.dataset.coco_common import coco_ann_to_shape_kwargs, parse_coco
from app.services.dataset.import_safety import UnsafeArchiveError, safe_extractall
from app.services.storage.factory import get_storage

# Decoding an image (local disk, fast) and pushing it to storage (a network
# round-trip on MinIO/prod) has no dependency on any other image in the
# zip, so staging them one at a time made a multi-hundred-image import
# scale linearly with image count for no reason. Staged concurrently, up to
# this many at a time; every `db` write (the `Image` row, its annotations)
# still happens on the caller's own thread — `Session` isn't thread-safe.
_IMPORT_MAX_WORKERS = 8

# A `db.commit()` is a full transaction sync; the `db.flush()` used mid-loop
# only needs to populate the new `Image` row's client-side UUID default
# before its annotations reference it by id. Committing every
# `_COMMIT_BATCH_SIZE` images (plus once more at the end) cuts commit count
# without changing what ends up persisted.
_COMMIT_BATCH_SIZE = 50


class CocoImportError(RuntimeError):
    pass


def _find_annotations_json(root: Path) -> Path:
    preferred = root / "annotations" / "instances_default.json"
    if preferred.exists():
        return preferred
    candidates = sorted(root.rglob("*.json"))
    if not candidates:
        raise CocoImportError("No .json file found in the uploaded zip")
    return candidates[0]


def _find_image_file(root: Path, file_name: str) -> Path | None:
    direct = root / "images" / file_name
    if direct.exists():
        return direct
    matches = list(root.rglob(Path(file_name).name))
    return matches[0] if matches else None


def _ensure_class_id(project: Project, name: str) -> int:
    existing = list(project.class_config or [])
    for entry in existing:
        if entry["name"] == name:
            return entry["id"]
    next_id = (max((entry["id"] for entry in existing), default=-1)) + 1
    existing.append({"id": next_id, "name": name})
    project.class_config = existing
    return next_id


def import_coco_zip(
    db: Session,
    *,
    project_id: uuid.UUID,
    zip_path: Path,
    dataset_name: str | None,
) -> Dataset:
    project = db.get(Project, project_id)
    if project is None:
        raise ValueError(f"No project with id {project_id}")

    if not zipfile.is_zipfile(zip_path):
        raise CocoImportError("Uploaded file is not a valid zip archive")

    dataset = Dataset(
        project_id=project_id,
        name=dataset_name or "coco-import",
        description="Imported from a COCO-format zip",
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    storage = get_storage()

    with tempfile.TemporaryDirectory() as tmp:
        extract_root = Path(tmp) / "extracted"
        extract_root.mkdir()
        with zipfile.ZipFile(zip_path) as zf:
            try:
                safe_extractall(zf, extract_root)
            except UnsafeArchiveError as exc:
                raise CocoImportError(str(exc)) from exc

        coco = json.loads(_find_annotations_json(extract_root).read_text(encoding="utf-8"))
        categories, annotations_by_image = parse_coco(coco)
        class_id_by_category: dict[int, int] = {
            cat_id: _ensure_class_id(project, name) for cat_id, name in categories.items()
        }

        def _stage_image(img_entry: dict) -> tuple[str, int, int] | None:
            """Runs off the main thread: local decode (for dimensions) plus
            the storage write, neither of which touches `db`. Returns
            `None` for a missing/undecodable image, same as the old inline
            `continue`."""
            src_path = _find_image_file(extract_root, img_entry["file_name"])
            if src_path is None:
                return None
            arr = cv2.imread(str(src_path))
            if arr is None:
                return None
            height, width = arr.shape[:2]
            width = img_entry.get("width") or width
            height = img_entry.get("height") or height
            key = safe_storage_key(
                str(project_id), str(dataset.id), "images", original_filename=img_entry["file_name"]
            )
            storage.upload(src_path, key, content_type="image/jpeg")
            return key, width, height

        img_entries = coco.get("images", [])
        staged: list[tuple[str, int, int] | None] = []
        if img_entries:
            max_workers = max(1, min(_IMPORT_MAX_WORKERS, len(img_entries)))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                # `.map` preserves input order in its results (each future
                # still runs concurrently; only the result iteration is
                # ordered) and re-raises the first exception it hits, same
                # as the old sequential loop aborting on the first failure.
                staged = list(pool.map(_stage_image, img_entries))

        for i, (img_entry, result) in enumerate(zip(img_entries, staged), start=1):
            if result is None:
                continue
            key, width, height = result

            image = Image(
                project_id=project_id,
                dataset_id=dataset.id,
                storage_key=key,
                original_filename=img_entry["file_name"],
                width=width,
                height=height,
                source_type=ImageSourceType.UPLOAD,
            )
            db.add(image)
            db.flush()

            for ann in annotations_by_image.get(img_entry["id"], []):
                class_id = class_id_by_category.get(ann["category_id"])
                if class_id is None:
                    continue
                create_annotation(
                    db,
                    image_id=image.id,
                    class_id=class_id,
                    class_name=categories[ann["category_id"]],
                    confidence=None,
                    source=AnnotationSource.HUMAN,
                    actor="coco-import",
                    **coco_ann_to_shape_kwargs(ann, width, height),
                )
            if i % _COMMIT_BATCH_SIZE == 0:
                db.commit()

    db.commit()
    return dataset
