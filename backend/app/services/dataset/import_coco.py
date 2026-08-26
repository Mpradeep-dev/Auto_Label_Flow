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
from pathlib import Path

import cv2
from sqlalchemy.orm import Session

from app.core.security import safe_storage_key
from app.models.annotation import AnnotationSource, ShapeType
from app.models.dataset import Dataset
from app.models.image import Image, ImageSourceType
from app.models.project import Project
from app.services.annotation.service import create_annotation
from app.services.dataset.import_safety import UnsafeArchiveError, safe_extractall
from app.services.storage.factory import get_storage


class CocoImportError(RuntimeError):
    pass


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _polygon_points_from_segmentation(segmentation, width: int, height: int) -> list[list[float]] | None:
    """COCO's `segmentation` is either a list of polygon rings
    (`[[x1,y1,x2,y2,...], ...]`) or an RLE dict (`{"counts": ..., "size": [h,w]}`).
    This app stores masks as polygons (see ShapeType docstring), so only the
    polygon-ring form is imported; RLE stays explicitly unsupported — the
    caller falls back to the bbox-only path for it, same as when
    `segmentation` is absent or empty. Only the first ring is used (single
    outer ring only, no holes — see PLAN non-goals)."""
    if not isinstance(segmentation, list) or not segmentation:
        return None
    ring = segmentation[0]
    if not isinstance(ring, list) or len(ring) < 6:  # need >=3 points * 2 coords
        return None
    flat = [float(v) for v in ring]
    points = [[_clamp01(flat[i] / width), _clamp01(flat[i + 1] / height)] for i in range(0, len(flat), 2)]
    return points if len(points) >= 3 else None


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
        categories = {c["id"]: c["name"] for c in coco.get("categories", [])}
        class_id_by_category: dict[int, int] = {
            cat_id: _ensure_class_id(project, name) for cat_id, name in categories.items()
        }

        annotations_by_image: dict[int, list[dict]] = {}
        for ann in coco.get("annotations", []):
            annotations_by_image.setdefault(ann["image_id"], []).append(ann)

        for img_entry in coco.get("images", []):
            src_path = _find_image_file(extract_root, img_entry["file_name"])
            if src_path is None:
                continue

            arr = cv2.imread(str(src_path))
            if arr is None:
                continue
            height, width = arr.shape[:2]
            width = img_entry.get("width") or width
            height = img_entry.get("height") or height

            key = safe_storage_key(
                str(project_id), str(dataset.id), "images", original_filename=img_entry["file_name"]
            )
            storage.upload(src_path, key, content_type="image/jpeg")

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
            db.commit()
            db.refresh(image)

            for ann in annotations_by_image.get(img_entry["id"], []):
                class_id = class_id_by_category.get(ann["category_id"])
                if class_id is None:
                    continue
                points = _polygon_points_from_segmentation(ann.get("segmentation"), width, height)
                if points is not None:
                    create_annotation(
                        db,
                        image_id=image.id,
                        class_id=class_id,
                        class_name=categories[ann["category_id"]],
                        shape_type=ShapeType.POLYGON,
                        points=points,
                        confidence=None,
                        source=AnnotationSource.HUMAN,
                        actor="coco-import",
                    )
                    continue
                bx, by, bw, bh = ann["bbox"]
                x1, y1, x2, y2 = bx / width, by / height, (bx + bw) / width, (by + bh) / height
                create_annotation(
                    db,
                    image_id=image.id,
                    class_id=class_id,
                    class_name=categories[ann["category_id"]],
                    x1=_clamp01(x1),
                    y1=_clamp01(y1),
                    x2=_clamp01(x2),
                    y2=_clamp01(y2),
                    confidence=None,
                    source=AnnotationSource.HUMAN,
                    actor="coco-import",
                )

    db.commit()
    return dataset
