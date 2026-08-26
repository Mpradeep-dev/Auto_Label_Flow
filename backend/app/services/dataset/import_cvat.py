"""Import a CVAT-XML ("CVAT for images 1.1") zip — the counterpart to
`export_cvat.py`. Same round-trip role as `import_coco.py`: export a CVAT
task as "CVAT for images 1.1", hand the zip here, get a new Dataset with
the same images and boxes.

Class identity is resolved by NAME against the project's existing
`class_config` (same rule as the Roboflow and COCO importers) — a CVAT
`label` name is just a string with no id to trust or distrust.
"""
from __future__ import annotations

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
from app.services.dataset.import_safety import UnsafeArchiveError, UnsafeXmlError, parse_xml_safely, safe_extractall
from app.services.storage.factory import get_storage


class CvatImportError(RuntimeError):
    pass


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _find_annotations_xml(root: Path) -> Path:
    direct = root / "annotations.xml"
    if direct.exists():
        return direct
    candidates = sorted(root.rglob("*.xml"))
    if not candidates:
        raise CvatImportError("No .xml file found in the uploaded zip")
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


def import_cvat_zip(
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
        raise CvatImportError("Uploaded file is not a valid zip archive")

    dataset = Dataset(
        project_id=project_id,
        name=dataset_name or "cvat-import",
        description="Imported from a CVAT-XML zip",
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
                raise CvatImportError(str(exc)) from exc

        try:
            xml_root = parse_xml_safely(_find_annotations_xml(extract_root))
        except UnsafeXmlError as exc:
            raise CvatImportError(str(exc)) from exc

        for image_el in xml_root.findall("image"):
            file_name = image_el.get("name")
            if not file_name:
                continue
            src_path = _find_image_file(extract_root, file_name)
            if src_path is None:
                continue

            arr = cv2.imread(str(src_path))
            if arr is None:
                continue
            height, width = arr.shape[:2]
            width = int(image_el.get("width") or width)
            height = int(image_el.get("height") or height)

            key = safe_storage_key(str(project_id), str(dataset.id), "images", original_filename=file_name)
            storage.upload(src_path, key, content_type="image/jpeg")

            image = Image(
                project_id=project_id,
                dataset_id=dataset.id,
                storage_key=key,
                original_filename=file_name,
                width=width,
                height=height,
                source_type=ImageSourceType.UPLOAD,
            )
            db.add(image)
            db.commit()
            db.refresh(image)

            for box_el in image_el.findall("box"):
                label = box_el.get("label")
                if not label:
                    continue
                class_id = _ensure_class_id(project, label)
                xtl, ytl = float(box_el.get("xtl", 0)), float(box_el.get("ytl", 0))
                xbr, ybr = float(box_el.get("xbr", 0)), float(box_el.get("ybr", 0))
                create_annotation(
                    db,
                    image_id=image.id,
                    class_id=class_id,
                    class_name=label,
                    x1=_clamp01(xtl / width),
                    y1=_clamp01(ytl / height),
                    x2=_clamp01(xbr / width),
                    y2=_clamp01(ybr / height),
                    confidence=None,
                    source=AnnotationSource.HUMAN,
                    actor="cvat-import",
                )

            for polygon_el in image_el.findall("polygon"):
                label = polygon_el.get("label")
                points_attr = polygon_el.get("points")
                if not label or not points_attr:
                    continue
                points: list[list[float]] = []
                for pair in points_attr.split(";"):
                    px_str, py_str = pair.split(",")
                    points.append([_clamp01(float(px_str) / width), _clamp01(float(py_str) / height)])
                if len(points) < 3:
                    continue
                class_id = _ensure_class_id(project, label)
                create_annotation(
                    db,
                    image_id=image.id,
                    class_id=class_id,
                    class_name=label,
                    shape_type=ShapeType.POLYGON,
                    points=points,
                    confidence=None,
                    source=AnnotationSource.HUMAN,
                    actor="cvat-import",
                )

    db.commit()
    return dataset
