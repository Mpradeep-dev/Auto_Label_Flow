"""CVAT-XML ("CVAT for images 1.1") export — the other format CVAT's own
UI imports directly (Task > Upload annotations > format "CVAT 1.1"), and
literally the format CVAT itself produces when you export a task that way.
Bounding boxes only (`<box>`) — this app doesn't have polygon/polyline
annotations to emit as CVAT's `<polygon>`/`<polyline>` yet.
"""
from __future__ import annotations

import shutil
import tempfile
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.security import safe_storage_key
from app.models.dataset import Dataset
from app.models.dataset_version import DatasetVersion
from app.services.dataset.version_data import VersionDataError, load_version_data, out_image_filename
from app.services.storage.factory import get_storage


class ExportError(RuntimeError):
    pass


def write_cvat_dataset(db: Session, *, version_id: uuid.UUID, root: Path) -> Path:
    """Materializes `images/` + `annotations.xml` under `root`. Returns the
    xml path."""
    try:
        data = load_version_data(db, version_id=version_id)
    except VersionDataError as exc:
        raise ExportError(str(exc)) from exc

    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    storage = get_storage()

    root_el = ET.Element("annotations")
    ET.SubElement(root_el, "version").text = "1.1"

    meta = ET.SubElement(root_el, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = data.dataset.name
    ET.SubElement(task, "size").text = str(len(data.images))
    ET.SubElement(task, "mode").text = "annotation"
    labels_el = ET.SubElement(task, "labels")
    for class_id in sorted(data.class_names):
        label_el = ET.SubElement(labels_el, "label")
        ET.SubElement(label_el, "name").text = data.class_names[class_id]

    for i, vi in enumerate(data.images):
        out_name = out_image_filename(vi.image)
        (images_dir / out_name).write_bytes(storage.read_bytes(vi.image.storage_key))

        image_el = ET.SubElement(
            root_el,
            "image",
            {
                "id": str(i),
                "name": out_name,
                "width": str(vi.image.width),
                "height": str(vi.image.height),
                # Not part of the CVAT-XML spec proper, but CVAT ignores
                # unknown attributes rather than rejecting them — carried
                # through so a re-import here can recover the split.
                "subset": vi.split,
            },
        )
        for event in vi.events:
            xtl = event.x1 * vi.image.width
            ytl = event.y1 * vi.image.height
            xbr = event.x2 * vi.image.width
            ybr = event.y2 * vi.image.height
            ET.SubElement(
                image_el,
                "box",
                {
                    "label": data.class_names.get(event.class_id, str(event.class_id)),
                    "xtl": f"{xtl:.2f}",
                    "ytl": f"{ytl:.2f}",
                    "xbr": f"{xbr:.2f}",
                    "ybr": f"{ybr:.2f}",
                    "occluded": "0",
                    "z_order": "0",
                },
            )

    ET.indent(root_el, space="  ")
    xml_bytes = ET.tostring(root_el, encoding="utf-8", xml_declaration=True)
    xml_path = root / "annotations.xml"
    xml_path.write_bytes(xml_bytes)
    if data.skipped_image_ids:
        (root / "SKIPPED_IMAGES.txt").write_text("\n".join(data.skipped_image_ids), encoding="utf-8")

    return xml_path


def export_cvat(db: Session, *, version_id: uuid.UUID) -> str:
    """Builds the CVAT-XML directory structure, zips it, uploads to storage,
    returns the storage key. Synchronous — same scale tradeoff as
    `export_yolo`."""
    version = db.get(DatasetVersion, version_id)
    if version is None:
        raise ExportError(f"No dataset version with id {version_id}")
    dataset = db.get(Dataset, version.dataset_id)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "dataset"
        root.mkdir()
        write_cvat_dataset(db, version_id=version_id, root=root)

        archive_base = str(Path(tmp) / "export")
        archive_path = shutil.make_archive(archive_base, "zip", root_dir=root)

        key = safe_storage_key(
            str(dataset.project_id),
            str(dataset.id),
            "exports",
            original_filename=f"{dataset.name}-v{version.version_number}-cvat.zip",
        )
        get_storage().upload(Path(archive_path), key, content_type="application/zip")

    return key
