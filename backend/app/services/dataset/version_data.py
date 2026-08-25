"""Shared "load one version's pinned images + annotations" reader, used by
every export format (YOLO, COCO, CVAT-XML) so they can never drift from
each other on what a version actually contains — each format module only
has to decide how to WRITE the data, not how to gather it. Reads the
pinned `AnnotationEvent` rows via the version's pins, never the live
`annotations` table (see `export_yolo.py`'s docstring for why: an export
must reproduce byte-identical labels regardless of review work done since).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.annotation import AnnotationEvent
from app.models.dataset import Dataset
from app.models.dataset_version import DatasetVersion, DatasetVersionAnnotationPin, DatasetVersionImage
from app.models.image import Image
from app.models.project import Project


class VersionDataError(RuntimeError):
    pass


@dataclass
class VersionImageData:
    image: Image
    split: str
    events: list[AnnotationEvent] = field(default_factory=list)


@dataclass
class VersionData:
    dataset: Dataset
    project: Project
    class_names: dict[int, str]  # class_id -> name
    images: list[VersionImageData]
    skipped_image_ids: list[str]


def load_version_data(db: Session, *, version_id: uuid.UUID) -> VersionData:
    version = db.get(DatasetVersion, version_id)
    if version is None:
        raise VersionDataError(f"No dataset version with id {version_id}")

    dataset = db.get(Dataset, version.dataset_id)
    project = db.get(Project, dataset.project_id)
    class_names = {entry["id"]: entry["name"] for entry in project.class_config}
    if not class_names:
        raise VersionDataError("Project has no class_config — register a model and set the project's classes first")

    version_images = list(
        db.scalars(select(DatasetVersionImage).where(DatasetVersionImage.dataset_version_id == version_id))
    )
    pins = list(
        db.scalars(
            select(DatasetVersionAnnotationPin).where(DatasetVersionAnnotationPin.dataset_version_id == version_id)
        )
    )
    event_ids = [pin.pinned_event_id for pin in pins]
    events_by_id = {
        e.id: e
        for e in (db.scalars(select(AnnotationEvent).where(AnnotationEvent.id.in_(event_ids))) if event_ids else [])
    }
    events_by_image: dict[uuid.UUID, list[AnnotationEvent]] = {}
    for pin in pins:
        event = events_by_id.get(pin.pinned_event_id)
        if event is not None:
            events_by_image.setdefault(pin.image_id, []).append(event)

    images: list[VersionImageData] = []
    skipped: list[str] = []
    for vi in version_images:
        image = db.get(Image, vi.image_id)
        if image is None:
            skipped.append(str(vi.image_id))  # deleted after being pinned — export what remains
            continue
        images.append(VersionImageData(image=image, split=vi.split.value, events=events_by_image.get(image.id, [])))

    return VersionData(dataset=dataset, project=project, class_names=class_names, images=images, skipped_image_ids=skipped)


def out_image_filename(image: Image) -> str:
    """Same naming every export format uses for a materialized image file —
    the image's own id, not its (untrusted, possibly-colliding) original
    filename."""
    ext = Path(image.original_filename).suffix or ".jpg"
    return f"{image.id}{ext}"
