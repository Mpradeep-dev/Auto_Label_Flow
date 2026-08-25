"""Pull an existing Roboflow project version in as a new Dataset (PLAN
follow-on: "import source", the other half of the Roboflow integration).

Imported boxes are treated as already-reviewed ground truth, not
model output: `source=HUMAN` (someone labeled them, just not in this app)
and the image is approved immediately via the existing `approve_image()`
service call — reusing it rather than setting `review_status` by hand
means the per-annotation status and the image status can never drift
apart (see `annotation/service.py`).

Class identity is resolved by NAME against the project's existing
`class_config`, never by raw index — Roboflow's own class ids are an
implementation detail of that one export and have no relationship to
whatever id a primary detector on this project already uses for the same
class name (PLAN "class taxonomy is read from the model, never hardcoded").
A name Roboflow has that the project doesn't gets appended with a new id.

`import_roboflow_raw_project`, below, is the fallback for a Roboflow
project that has never had a Version generated — common for a project
someone is still uploading/labeling in Roboflow's own Annotate tab. A
Version is the only thing `.download()` can pull, so that path is a dead
end for a 0-version project; this one reads the same raw images and boxes
straight off the project's `search()`/`image()` endpoints instead.
"""
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import requests
import yaml
from sqlalchemy.orm import Session

from app.core.security import safe_storage_key
from app.models.annotation import AnnotationSource
from app.models.dataset import Dataset
from app.models.image import Image, ImageSourceType
from app.models.project import Project
from app.services.annotation.service import approve_image, create_annotation
from app.services.integrations.roboflow_connect import get_client
from app.services.storage.factory import get_storage

_SPLIT_DIRS = ("train", "valid", "test")
_RAW_SEARCH_PAGE_SIZE = 100


def _merge_class_config(project: Project, roboflow_names: list[str]) -> dict[int, tuple[int, str]]:
    """Returns {roboflow_class_index: (project_class_id, class_name)}.
    Extends `project.class_config` in place (caller commits) for any name
    Roboflow has that the project doesn't yet."""
    existing = list(project.class_config or [])
    by_name = {entry["name"]: entry["id"] for entry in existing}
    next_id = (max((entry["id"] for entry in existing), default=-1)) + 1

    mapping: dict[int, tuple[int, str]] = {}
    for rf_idx, name in enumerate(roboflow_names):
        if name in by_name:
            mapping[rf_idx] = (by_name[name], name)
        else:
            existing.append({"id": next_id, "name": name})
            by_name[name] = next_id
            mapping[rf_idx] = (next_id, name)
            next_id += 1

    project.class_config = existing
    return mapping


def import_roboflow_project(
    db: Session,
    *,
    project_id: uuid.UUID,
    workspace: str,
    project_slug: str,
    version: int,
    dataset_name: str | None,
    progress_cb: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Dataset:
    """`progress_cb(current, total)`, if given, is called once with
    `current=0` as soon as the image count is known (download finished),
    then once per image as it's persisted. `should_cancel()` is checked
    before each image, stopping the loop early (partial dataset kept, not
    rolled back) — and once more before `.version().download()`, the one
    step in here a cancel can't interrupt once started: it's a single
    blocking SDK call that pulls and unzips the whole version, with no
    hook to check a flag partway through. Checking first at least catches
    "cancelled before the heavy part ever began" instead of doing nothing
    until it finishes regardless."""
    project = db.get(Project, project_id)
    if project is None:
        raise ValueError(f"No project with id {project_id}")

    if should_cancel is not None and should_cancel():
        dataset = Dataset(
            project_id=project_id,
            name=dataset_name or f"roboflow-{workspace}-{project_slug}-v{version}",
            description=f"Cancelled before pulling from Roboflow {workspace}/{project_slug} version {version}.",
        )
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        if progress_cb is not None:
            progress_cb(0, 0)
        return dataset

    rf, _config = get_client(db)
    rf_project = rf.workspace(workspace).project(project_slug)
    rf_version = rf_project.version(version)

    with tempfile.TemporaryDirectory() as tmp:
        rf_dataset = rf_version.download("yolov8", location=str(Path(tmp) / "download"))
        location = Path(rf_dataset.location)

        data_yaml = yaml.safe_load((location / "data.yaml").read_text(encoding="utf-8"))
        names = data_yaml.get("names", [])
        if isinstance(names, dict):
            names = [names[i] for i in sorted(names, key=int)]
        class_mapping = _merge_class_config(project, names)

        dataset = Dataset(
            project_id=project_id,
            name=dataset_name or f"roboflow-{workspace}-{project_slug}-v{version}",
            description=f"Imported from Roboflow {workspace}/{project_slug} version {version}",
        )
        db.add(dataset)
        db.commit()
        db.refresh(dataset)

        storage = get_storage()

        # Collected up front (not processed lazily split-by-split) so the
        # total image count is known before the first progress_cb call —
        # without it the caller has nothing to size a progress bar against.
        pending: list[tuple[str, Path]] = []
        for split in _SPLIT_DIRS:
            images_dir = location / split / "images"
            if not images_dir.is_dir():
                continue
            for image_path in sorted(images_dir.glob("*")):
                if image_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    pending.append((split, image_path))

        total = len(pending)
        if progress_cb is not None:
            progress_cb(0, total)

        for i, (split, image_path) in enumerate(pending):
            if should_cancel is not None and should_cancel():
                break

            labels_dir = location / split / "labels"
            img = cv2.imread(str(image_path))
            if img is None:
                if progress_cb is not None:
                    progress_cb(i + 1, total)
                continue
            height, width = img.shape[:2]

            key = safe_storage_key(
                str(project_id), str(dataset.id), "images", original_filename=image_path.name
            )
            storage.upload(image_path, key, content_type="image/jpeg")

            image = Image(
                project_id=project_id,
                dataset_id=dataset.id,
                storage_key=key,
                original_filename=image_path.name,
                width=width,
                height=height,
                source_type=ImageSourceType.UPLOAD,
            )
            db.add(image)
            db.commit()
            db.refresh(image)

            label_path = labels_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    rf_class_idx, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
                    if rf_class_idx not in class_mapping:
                        continue
                    class_id, class_name = class_mapping[rf_class_idx]
                    create_annotation(
                        db,
                        image_id=image.id,
                        class_id=class_id,
                        class_name=class_name,
                        x1=cx - w / 2,
                        y1=cy - h / 2,
                        x2=cx + w / 2,
                        y2=cy + h / 2,
                        confidence=None,
                        source=AnnotationSource.HUMAN,
                        actor="roboflow-import",
                    )
            approve_image(db, image_id=image.id)

            if progress_cb is not None:
                progress_cb(i + 1, total)

        db.commit()

    return dataset


def _ensure_class_id(project: Project, name: str) -> int:
    """Same by-name resolution as `_merge_class_config`, just incremental —
    the raw path discovers class names one box at a time (off each image's
    own annotation) rather than all at once from a `data.yaml`."""
    existing = list(project.class_config or [])
    for entry in existing:
        if entry["name"] == name:
            return entry["id"]
    next_id = (max((entry["id"] for entry in existing), default=-1)) + 1
    existing.append({"id": next_id, "name": name})
    project.class_config = existing
    return next_id


def import_roboflow_raw_project(
    db: Session,
    *,
    project_id: uuid.UUID,
    workspace: str,
    project_slug: str,
    dataset_name: str | None,
    unannotated_only: bool = False,
    progress_cb: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Dataset:
    """Pulls a project's raw uploaded images directly, for a project with
    no generated Version to `.download()`. Images are left PENDING, not
    auto-approved like `import_roboflow_project`: a raw Roboflow upload
    isn't a curated export, so whatever boxes exist on it (there may be
    none at all) aren't treated as already-reviewed ground truth here —
    they're a starting point to review in this app instead.

    `unannotated_only` narrows the pull to images with zero existing
    Roboflow annotations — useful when the point of pulling them in here
    is specifically to label what nobody's touched yet, rather than
    re-reviewing images Roboflow already has boxes on. Filtered locally
    off each item's `annotations.count` (the `search()` endpoint has no
    server-side "has no annotations" filter to push this down to).

    `should_cancel()` is checked both between search pages (this path has
    no single giant blocking call the way a version `.download()` does, so
    even the listing phase can stop early) and before each image."""
    project = db.get(Project, project_id)
    if project is None:
        raise ValueError(f"No project with id {project_id}")

    rf, _config = get_client(db)
    rf_project = rf.workspace(workspace).project(project_slug)

    items: list[dict] = []
    offset = 0
    while True:
        if should_cancel is not None and should_cancel():
            break
        page = rf_project.search(
            offset=offset, limit=_RAW_SEARCH_PAGE_SIZE, fields=["id", "name", "annotations"]
        )
        items.extend(page)
        if len(page) < _RAW_SEARCH_PAGE_SIZE:
            break
        offset += _RAW_SEARCH_PAGE_SIZE

    if unannotated_only:
        items = [item for item in items if (item.get("annotations") or {}).get("count", 0) == 0]

    total = len(items)
    if progress_cb is not None:
        progress_cb(0, total)

    dataset = Dataset(
        project_id=project_id,
        name=dataset_name or f"roboflow-{workspace}-{project_slug}-raw",
        description=f"Imported (raw, unversioned) from Roboflow {workspace}/{project_slug}",
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    storage = get_storage()

    for i, item in enumerate(items):
        if should_cancel is not None and should_cancel():
            break

        details = rf_project.image(item["id"])
        url = (details.get("urls") or {}).get("original")
        if not url:
            if progress_cb is not None:
                progress_cb(i + 1, total)
            continue

        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        arr = cv2.imdecode(np.frombuffer(resp.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            if progress_cb is not None:
                progress_cb(i + 1, total)
            continue
        height, width = arr.shape[:2]

        original_filename = details.get("name") or f"{item['id']}.jpg"
        key = safe_storage_key(str(project_id), str(dataset.id), "images", original_filename=original_filename)
        storage.upload_bytes(resp.content, key, content_type="image/jpeg")

        image = Image(
            project_id=project_id,
            dataset_id=dataset.id,
            storage_key=key,
            original_filename=original_filename,
            width=width,
            height=height,
            source_type=ImageSourceType.UPLOAD,
        )
        db.add(image)
        db.commit()
        db.refresh(image)

        annotation = details.get("annotation") or {}
        ann_w = annotation.get("width") or width
        ann_h = annotation.get("height") or height
        for box in annotation.get("boxes") or []:
            class_id = _ensure_class_id(project, box["label"])
            cx = float(box["x"]) / ann_w
            cy = float(box["y"]) / ann_h
            w = float(box["width"]) / ann_w
            h = float(box["height"]) / ann_h
            create_annotation(
                db,
                image_id=image.id,
                class_id=class_id,
                class_name=box["label"],
                x1=cx - w / 2,
                y1=cy - h / 2,
                x2=cx + w / 2,
                y2=cy + h / 2,
                confidence=None,
                source=AnnotationSource.HUMAN,
                actor="roboflow-import",
            )

        if progress_cb is not None:
            progress_cb(i + 1, total)

    db.commit()
    return dataset
