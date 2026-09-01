"""Register images that already live in the app's Azure Blob container as
`Image` rows **by reference** — no byte copy, unlike `roboflow_import.py`
or `import_coco.py`, which both download every image and re-upload a
second copy under this app's own `project/<id>/dataset/<id>/images/`
layout. At tens of thousands of already-labelled production images that
duplication is the whole problem this path avoids: each `Image.storage_key`
here points straight at the existing blob, and `is_external=True` marks
the row so deleting it never deletes someone else's blob.

Scope (matches the approved design):
  - Images live in the app's configured `AZURE_STORAGE_CONTAINER`, under a
    prefix — so `AzureBlobStorage.get_url()` SAS-signs them unchanged and
    the annotation canvas streams them with no local copy.
  - Labels sit alongside as **YOLO** (`data.yaml` + `labels/<stem>.txt`,
    normalised `cls cx cy w h`) or **COCO** (`*.json` with
    `images`/`annotations`/`categories`); `label_format="auto"` detects
    which.
  - Imported images are approved immediately as ground truth (like the
    Roboflow *version* importer), `source=HUMAN`.
  - Class identity is resolved by NAME against the project's
    `class_config`, never by raw index — same rule as every other
    importer.

This module is legitimately Azure-specific (as `roboflow_import.py` is
Roboflow-SDK-specific), so it builds its own `ContainerClient` for
listing + header reads and leaves the `ObjectStorage` interface untouched;
`get_storage()` is only involved later, at serve time, and already works
because the key is a real blob name.
"""
from __future__ import annotations

import json
import logging
import posixpath
import uuid
from dataclasses import dataclass
from io import BytesIO
from typing import Callable

import yaml
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.annotation import AnnotationSource
from app.models.dataset import Dataset
from app.models.image import Image, ImageSourceType
from app.models.project import Project
from app.services.annotation.service import approve_image, create_annotation
from app.services.dataset.coco_common import clamp01, coco_ann_to_shape_kwargs, parse_coco

logger = logging.getLogger(__name__)

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
_HEADER_BYTES = 65536  # enough for JPEG/PNG dimension headers without decoding pixels


class AzureBlobImportError(RuntimeError):
    pass


@dataclass
class _Pending:
    """One image to register: its blob name, dimensions if a COCO entry
    already gave them (else read from the header lazily), and a closure
    that writes this image's annotations once its `Image.id` exists."""

    blob: str
    width: int | None
    height: int | None
    make_annotations: Callable[[Session, uuid.UUID, int, int], None]


def _ensure_class_id(project: Project, name: str) -> int:
    """By-NAME class resolution against `project.class_config`, extending it
    in place for an unseen name — identical rule to `import_coco.py` /
    `roboflow_import.py` (a raw external index is only meaningful within the
    export that produced it)."""
    existing = list(project.class_config or [])
    for entry in existing:
        if entry["name"] == name:
            return entry["id"]
    next_id = (max((entry["id"] for entry in existing), default=-1)) + 1
    existing.append({"id": next_id, "name": name})
    project.class_config = existing
    return next_id


def _container_client():
    from azure.storage.blob import ContainerClient  # local import: Azure stays optional at dev time

    if settings.STORAGE_BACKEND != "azure" or not settings.AZURE_STORAGE_CONNECTION_STRING:
        raise AzureBlobImportError(
            "Azure Blob import requires STORAGE_BACKEND=azure and AZURE_STORAGE_CONNECTION_STRING"
        )
    return ContainerClient.from_connection_string(
        settings.AZURE_STORAGE_CONNECTION_STRING, settings.AZURE_STORAGE_CONTAINER
    )


def _read_dims(container, blob_name: str) -> tuple[int, int]:
    """`(width, height)` from the blob's header only — a ranged read of the
    first `_HEADER_BYTES`, parsed by Pillow (which reads size from the
    header without decoding). Falls back to a full read on the rare image
    whose header spills past the slice."""
    from PIL import Image as PILImage

    head = container.download_blob(blob_name, offset=0, length=_HEADER_BYTES).readall()
    try:
        with PILImage.open(BytesIO(head)) as im:
            return int(im.size[0]), int(im.size[1])
    except Exception:
        full = container.download_blob(blob_name).readall()
        with PILImage.open(BytesIO(full)) as im:
            return int(im.size[0]), int(im.size[1])


def _list_blob_names(container, prefix: str) -> list[str]:
    return [b.name for b in container.list_blobs(name_starts_with=prefix)]


def _find_coco_doc(container, blob_names: list[str]) -> dict | None:
    for name in blob_names:
        if not name.lower().endswith(".json"):
            continue
        try:
            doc = json.loads(container.download_blob(name).readall())
        except ValueError:
            continue
        if isinstance(doc, dict) and "images" in doc and "annotations" in doc:
            return doc
    return None


def _detect_format(container, blob_names: list[str], prefix: str) -> str:
    if any(posixpath.basename(b) == "data.yaml" for b in blob_names) or any(
        b.lower().endswith(".txt") for b in blob_names
    ):
        return "yolo"
    if _find_coco_doc(container, blob_names) is not None:
        return "coco"
    raise AzureBlobImportError(
        f"Could not detect YOLO (data.yaml / .txt) or COCO (.json) labels under prefix {prefix!r}"
    )


def _load_yolo_class_names(container, blob_names: list[str]) -> list[str]:
    for name in blob_names:
        if posixpath.basename(name) != "data.yaml":
            continue
        data = yaml.safe_load(container.download_blob(name).readall().decode("utf-8")) or {}
        names = data.get("names", [])
        if isinstance(names, dict):
            names = [names[i] for i in sorted(names, key=int)]
        return list(names)
    return []


def _find_label_blob(image_blob: str, blob_set: set[str]) -> str | None:
    """The YOLO `.txt` for an image blob: a same-directory sibling, or the
    `images/` → `labels/` swap Roboflow's export uses."""
    same_dir = posixpath.splitext(image_blob)[0] + ".txt"
    if same_dir in blob_set:
        return same_dir
    if "/images/" in image_blob:
        swapped = posixpath.splitext(image_blob.replace("/images/", "/labels/", 1))[0] + ".txt"
        if swapped in blob_set:
            return swapped
    d, base = posixpath.split(image_blob)
    if posixpath.basename(d) == "images":
        swapped = posixpath.join(posixpath.dirname(d), "labels", posixpath.splitext(base)[0] + ".txt")
        if swapped in blob_set:
            return swapped
    return None


def _parse_yolo_txt(raw: str) -> list[tuple[int, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float]] = []
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            ci = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            continue
        rows.append((ci, cx, cy, w, h))
    return rows


def _yolo_pending(
    container, image_blobs: list[str], blob_set: set[str], class_names: list[str], project: Project
) -> list[_Pending]:
    pending: list[_Pending] = []
    for ib in image_blobs:
        label_blob = _find_label_blob(ib, blob_set)
        raw = container.download_blob(label_blob).readall().decode("utf-8") if label_blob else ""
        rows = _parse_yolo_txt(raw)

        def make(db: Session, image_id: uuid.UUID, _w: int, _h: int, rows=rows) -> None:
            for ci, cx, cy, w, h in rows:
                name = class_names[ci] if 0 <= ci < len(class_names) else str(ci)
                create_annotation(
                    db,
                    image_id=image_id,
                    class_id=_ensure_class_id(project, name),
                    class_name=name,
                    x1=clamp01(cx - w / 2),
                    y1=clamp01(cy - h / 2),
                    x2=clamp01(cx + w / 2),
                    y2=clamp01(cy + h / 2),
                    confidence=None,
                    source=AnnotationSource.HUMAN,
                    actor="azure-blob-import",
                )

        pending.append(_Pending(ib, None, None, make))
    return pending


def _coco_pending(coco: dict, image_blobs: list[str], project: Project) -> list[_Pending]:
    categories, anns_by_image = parse_coco(coco)
    by_basename = {posixpath.basename(b): b for b in image_blobs}

    pending: list[_Pending] = []
    for img_entry in coco.get("images", []):
        blob = by_basename.get(posixpath.basename(img_entry["file_name"]))
        if blob is None:
            continue
        anns = anns_by_image.get(img_entry["id"], [])

        def make(db: Session, image_id: uuid.UUID, w: int, h: int, anns=anns) -> None:
            for ann in anns:
                name = categories.get(ann["category_id"])
                if name is None:
                    continue
                create_annotation(
                    db,
                    image_id=image_id,
                    class_id=_ensure_class_id(project, name),
                    class_name=name,
                    confidence=None,
                    source=AnnotationSource.HUMAN,
                    actor="azure-blob-import",
                    **coco_ann_to_shape_kwargs(ann, w, h),
                )

        pending.append(
            _Pending(blob, img_entry.get("width"), img_entry.get("height"), make)
        )
    return pending


def import_azure_blob_prefix(
    db: Session,
    *,
    project_id: uuid.UUID,
    prefix: str,
    label_format: str = "auto",
    dataset_name: str | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Dataset:
    """Walk `prefix` in the app's Azure container, create one `Image` row
    per image blob (referencing it in place), attach its YOLO/COCO labels
    as approved `HUMAN` annotations.

    `progress_cb(current, total)` fires once with `current=0` when the blob
    listing is done, then once per image. `should_cancel()` is checked
    before the loop and before each image; a partial dataset is kept, not
    rolled back (matching `roboflow_import.py`)."""
    project = db.get(Project, project_id)
    if project is None:
        raise ValueError(f"No project with id {project_id}")

    container = _container_client()
    blob_names = _list_blob_names(container, prefix)
    image_blobs = sorted(b for b in blob_names if posixpath.splitext(b)[1].lower() in _IMAGE_EXTS)

    resolved = label_format if label_format != "auto" else _detect_format(container, blob_names, prefix)

    dataset = Dataset(
        project_id=project_id,
        name=dataset_name or f"azure-blob-{(prefix.strip('/') or 'root').replace('/', '-')}",
        description=f"Referenced in place from Azure Blob prefix {prefix!r} ({resolved} labels)",
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    if resolved == "yolo":
        class_names = _load_yolo_class_names(container, blob_names)
        if not class_names:
            raise AzureBlobImportError(f"No data.yaml with a class list found under prefix {prefix!r}")
        pending = _yolo_pending(container, image_blobs, set(blob_names), class_names, project)
    elif resolved == "coco":
        coco = _find_coco_doc(container, blob_names)
        if coco is None:
            raise AzureBlobImportError(f"No COCO-shaped .json found under prefix {prefix!r}")
        pending = _coco_pending(coco, image_blobs, project)
    else:
        raise AzureBlobImportError(f"Unknown label_format {label_format!r} (expected auto, yolo or coco)")

    total = len(pending)
    if progress_cb is not None:
        progress_cb(0, total)

    for i, item in enumerate(pending):
        if should_cancel is not None and should_cancel():
            break

        width, height = item.width, item.height
        if not width or not height:
            width, height = _read_dims(container, item.blob)

        image = Image(
            project_id=project_id,
            dataset_id=dataset.id,
            storage_key=item.blob,  # the existing blob, verbatim — no re-key, no upload
            original_filename=posixpath.basename(item.blob),
            width=width,
            height=height,
            is_external=True,
            source_type=ImageSourceType.UPLOAD,
        )
        db.add(image)
        db.flush()  # need image.id for the annotation writes below

        item.make_annotations(db, image.id, width, height)
        approve_image(db, image_id=image.id)  # commits; images are curated ground truth

        if progress_cb is not None:
            progress_cb(i + 1, total)

    db.commit()
    return dataset
