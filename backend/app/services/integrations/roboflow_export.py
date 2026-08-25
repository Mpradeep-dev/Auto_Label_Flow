"""Push a dataset version straight to a Roboflow project (PLAN follow-on:
"export destination" + "import source", both directions).

Reuses `write_yolo_dataset()` (the same materialization `export_yolo()` zips
for download and `local_provider.py` trains from) so the images and labels
pushed here are byte-identical to what a local export or a local training
run would see — one source of truth for "what does this dataset version
actually contain."

Runs synchronously in the request, same tradeoff `export_yolo()` already
makes (see its docstring): fine at this project's dataset scale, would move
to a Celery task without changing this logic if that stops being true.
"""
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Callable

from sqlalchemy.orm import Session

from app.services.dataset.export_yolo import ExportError, write_yolo_dataset
from app.services.integrations.roboflow_connect import get_client

# Roboflow's own convention names the validation split "valid"; our export
# (and YOLO's) calls it "val" — map at the upload boundary only, everything
# upstream keeps using "val" (see splitter.py / export_yolo.py SplitName).
_SPLIT_TO_ROBOFLOW = {"train": "train", "val": "valid", "test": "test"}


def push_version_to_roboflow(
    db: Session,
    *,
    version_id: uuid.UUID,
    workspace: str,
    project_slug: str,
    progress_cb: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[int, int, list[str]]:
    """Returns (uploaded_count, failed_count, failure_messages).

    `progress_cb(current, total)`, if given, is called once with
    `current=0` as soon as the image count is known (materialization
    finished), then once per image after its upload attempt (success or
    failure alike). `should_cancel()`, checked before each image, stops the
    loop early — whatever's already uploaded to Roboflow stays uploaded,
    same as a cancel partway through any other batch job here."""
    if should_cancel is not None and should_cancel():
        return 0, 0, []

    rf, _config = get_client(db)
    project = rf.workspace(workspace).project(project_slug)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "dataset"
        root.mkdir()
        try:
            data_yaml_path = write_yolo_dataset(db, version_id=version_id, root=root)
        except ExportError:
            raise

        pending: list[tuple[str, Path]] = []
        for split in _SPLIT_TO_ROBOFLOW:
            images_dir = root / "images" / split
            pending.extend((split, p) for p in sorted(images_dir.glob("*")))

        total = len(pending)
        if progress_cb is not None:
            progress_cb(0, total)

        uploaded = 0
        failed = 0
        failures: list[str] = []
        for i, (split, image_path) in enumerate(pending):
            if should_cancel is not None and should_cancel():
                break

            roboflow_split = _SPLIT_TO_ROBOFLOW[split]
            labels_dir = root / "labels" / split
            label_path = labels_dir / f"{image_path.stem}.txt"
            try:
                project.upload(
                    image_path=str(image_path),
                    annotation_path=str(label_path) if label_path.exists() else None,
                    annotation_labelmap=str(data_yaml_path),
                    split=roboflow_split,
                )
                uploaded += 1
            except Exception as exc:  # a single bad image shouldn't abort the whole push
                failed += 1
                failures.append(f"{image_path.name}: {exc}")

            if progress_cb is not None:
                progress_cb(i + 1, total)

    return uploaded, failed, failures
