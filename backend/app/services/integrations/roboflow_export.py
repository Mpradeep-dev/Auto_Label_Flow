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

import logging
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable

from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.models.dataset_version import DatasetVersion
from app.services.dataset.export_yolo import ExportError, write_yolo_dataset
from app.services.integrations.roboflow_connect import get_client

logger = logging.getLogger(__name__)

# Roboflow's own convention names the validation split "valid"; our export
# (and YOLO's) calls it "val" — map at the upload boundary only, everything
# upstream keeps using "val" (see splitter.py / export_yolo.py SplitName).
_SPLIT_TO_ROBOFLOW = {"train": "train", "val": "valid", "test": "test"}

# Roboflow's upload endpoint only accepts a batch identifier matching
# `^[a-z0-9_-]{1,64}$` — no spaces, no uppercase, no punctuation. An upload
# carrying a `batch` value outside that set is dropped server-side: the
# image never lands in the Annotate tab (or anywhere else) and the SDK
# raises, so the count shows up under `failed`, not `uploaded`. A dataset
# name is free text (spaces, caps, parens all allowed here), so the
# `AutoLabelFlow-{dataset}-v{n}` label MUST be slugified before it goes on
# the wire. This mirrors what Roboflow's own UI does to a batch name typed
# with spaces/caps.
_BATCH_NAME_MAX_LEN = 64
_BATCH_NAME_FORBIDDEN = re.compile(r"[^a-z0-9_-]+")


def _sanitize_batch_name(raw: str) -> str:
    """Coerce `raw` into Roboflow's `^[a-z0-9_-]{1,64}$` rule: lowercase,
    every run of other characters collapsed to a single '-', ends trimmed
    of '-'/'_', length capped. Falls back to a fixed label if nothing
    usable survives (e.g. a dataset named only in a non-Latin script)."""
    slug = _BATCH_NAME_FORBIDDEN.sub("-", raw.lower()).strip("-_")
    slug = slug[:_BATCH_NAME_MAX_LEN].strip("-_")
    return slug or "autolabelflow"


# Roboflow's upload endpoint intermittently answers with a transient 5xx/429
# — a bare "500 Server Error / try again in 30 seconds" page from Google
# Frontend, not a Roboflow JSON error — the same failure mode
# `roboflow_import._rf_search_page` already retries. Retry those per image
# with short backoff (1s, 2s); a 4xx is not transient and fails that image
# at once.
_UPLOAD_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_UPLOAD_MAX_ATTEMPTS = 3
_UPLOAD_BACKOFF_BASE_S = 1.0
_AUTH_STATUSES = frozenset({401, 403})

# When nothing has uploaded yet and the run has already failed this many
# images the *same* way, stop. Grinding through a multi-thousand-image
# version at ~10s per failed attempt is a >10h "RUNNING" job that uploads
# nothing — over-quota / expired-plan / wrong-key / Roboflow-down all look
# like this, and none are fixed by trying the next image.
_FAIL_FAST_AFTER = 5


class RoboflowExportError(RuntimeError):
    """Abort a push that more retries won't fix — Roboflow returning 5xx for
    every image (quota / plan / incident) or rejecting the key. The message
    is written for the user and surfaced verbatim on the job row by
    `run_roboflow_export`."""


def _describe_upload_error(exc: Exception) -> str:
    """`ImageUploadError` (and its `RoboflowError` base) carry an HTTP
    `status_code` and `message`; anything else just stringifies."""
    status = getattr(exc, "status_code", None)
    detail = str(getattr(exc, "message", None) or exc).strip()[:300]
    return f"HTTP {status}: {detail}" if status is not None else detail


def _fail_fast_message(statuses: list[int | None], failures: list[str]) -> str:
    """Turn the first `_FAIL_FAST_AFTER` failures into one actionable
    sentence for the job row, tailored to what the status codes say."""
    n = len(failures)
    codes = sorted({s for s in statuses if s is not None})
    if codes and all(c in _UPLOAD_RETRY_STATUSES for c in codes):
        return (
            f"Roboflow returned a server error (HTTP {codes}) for the first {n} uploads "
            "and nothing was pushed, so the export stopped instead of retrying every "
            "remaining image. This is almost always an exhausted monthly upload quota "
            "or an expired plan on the Roboflow workspace, a target project that is "
            "unavailable, or a Roboflow incident (check status.roboflow.com). There is "
            "no way to push past a quota limit from here — raise the plan limit, free "
            "up room, or point the export at a workspace with capacity, then re-run. To "
            "export without Roboflow, use the Export page to download the version as "
            "YOLO, COCO, or CVAT."
        )
    if any(c in _AUTH_STATUSES for c in codes):
        return (
            f"Roboflow rejected the first {n} uploads as unauthorized (HTTP {codes}). "
            "Check the connected key is a Roboflow Private API Key with write access to "
            "this workspace (Settings -> Roboflow), then re-run."
        )
    first = failures[0] if failures else "unknown error"
    return (
        f"The first {n} uploads to Roboflow all failed and nothing was pushed, so the "
        f"export stopped. First error - {first}. Fix the cause and re-run, or use the "
        "Export page to download the version instead."
    )


def _upload_one_image(project, **upload_kwargs) -> None:
    """`project.upload(**upload_kwargs)` with a bounded retry on a transient
    5xx/429. Re-raises the last error once attempts are exhausted, and
    immediately for any non-transient status."""
    for attempt in range(1, _UPLOAD_MAX_ATTEMPTS + 1):
        try:
            project.upload(**upload_kwargs)
            return
        except Exception as exc:  # noqa: BLE001 - re-raised below, classified by status
            status = getattr(exc, "status_code", None)
            if status not in _UPLOAD_RETRY_STATUSES or attempt == _UPLOAD_MAX_ATTEMPTS:
                raise
            logger.warning(
                "Roboflow export: %s failed HTTP %s (attempt %d/%d) — retrying",
                upload_kwargs.get("image_path"),
                status,
                attempt,
                _UPLOAD_MAX_ATTEMPTS,
            )
            time.sleep(_UPLOAD_BACKOFF_BASE_S * 2 ** (attempt - 1))


def push_version_to_roboflow(
    db: Session,
    *,
    version_id: uuid.UUID,
    workspace: str,
    project_slug: str,
    progress_cb: Callable[[int, int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[int, int, list[str]]:
    """Returns (uploaded_count, failed_count, failure_messages).

    `progress_cb(uploaded, total, failed)`, if given, is called once with
    `uploaded=0` as soon as the image count is known (materialization
    finished), then once per image after its upload attempt. `uploaded`
    counts only images that actually reached Roboflow — a run where every
    image fails never advances it, so the UI shows "0 uploaded, N failed"
    rather than a bar creeping forward on work that didn't land.
    `should_cancel()`, checked before each image, stops the loop early —
    whatever's already uploaded to Roboflow stays uploaded, same as a
    cancel partway through any other batch job here."""
    if should_cancel is not None and should_cancel():
        return 0, 0, []

    rf, _config = get_client(db)
    project = rf.workspace(workspace).project(project_slug)

    # Left unset, the SDK groups every upload under its own hardcoded
    # `DEFAULT_BATCH_NAME` ("Pip Package Upload") — meaningless in
    # Roboflow's UI once more than one project or app pushes into the same
    # Roboflow project. Name the batch after this app and the dataset
    # version it came from instead, so it's identifiable at a glance.
    version = db.get(DatasetVersion, version_id)
    dataset = db.get(Dataset, version.dataset_id) if version is not None else None
    # `{dataset.name}-v{version_number}` matches the naming already used for
    # this version's own export filenames (export_yolo.py/export_coco.py/
    # export_cvat.py) — same identifier, just also visible in Roboflow now.
    # Slugified before use: Roboflow silently drops uploads whose `batch`
    # isn't `^[a-z0-9_-]{1,64}$` (see `_sanitize_batch_name`).
    raw_batch_name = (
        f"AutoLabelFlow-{dataset.name}-v{version.version_number}"
        if version is not None and dataset is not None
        else "AutoLabelFlow"
    )
    batch_name = _sanitize_batch_name(raw_batch_name)
    logger.info(
        "Roboflow export: version=%s dataset=%r -> %s/%s batch=%r",
        version_id,
        getattr(dataset, "name", None),
        workspace,
        project_slug,
        batch_name,
    )

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
            progress_cb(0, total, 0)

        uploaded = 0
        failed = 0
        failures: list[str] = []
        seen_statuses: list[int | None] = []
        for i, (split, image_path) in enumerate(pending):
            if should_cancel is not None and should_cancel():
                break

            roboflow_split = _SPLIT_TO_ROBOFLOW[split]
            labels_dir = root / "labels" / split
            label_path = labels_dir / f"{image_path.stem}.txt"
            # `write_yolo_dataset` writes a labels/*.txt for every image,
            # including one with zero annotations — for YOLO/training that's
            # the standard "no objects" convention, but the file exists and
            # is merely empty, not absent. Roboflow's own parser can't
            # recognize an empty YOLO annotation file (HTTP 400
            # "Unrecognized annotation format" — confirmed 1:1 against every
            # unannotated image in a real push) and there's no annotation to
            # lose by omitting it, so treat empty the same as missing here.
            has_annotation = label_path.exists() and label_path.stat().st_size > 0
            try:
                _upload_one_image(
                    project,
                    image_path=str(image_path),
                    annotation_path=str(label_path) if has_annotation else None,
                    annotation_labelmap=str(data_yaml_path),
                    split=roboflow_split,
                    batch_name=batch_name,
                )
                uploaded += 1
            except Exception as exc:  # a single bad image shouldn't abort the whole push
                failed += 1
                seen_statuses.append(getattr(exc, "status_code", None))
                detail = _describe_upload_error(exc)
                failures.append(f"{image_path.name}: {detail}")
                logger.warning(
                    "Roboflow export: upload failed for %s (%s)", image_path.name, detail, exc_info=True
                )
                # Systemic failure: nothing has landed and the first N
                # images all failed. Retrying the rest one-by-one for hours
                # won't help — stop with a message that names the likely
                # cause (`run_roboflow_export` puts it on the job row).
                if uploaded == 0 and failed >= _FAIL_FAST_AFTER:
                    raise RoboflowExportError(_fail_fast_message(seen_statuses, failures)) from exc

            if progress_cb is not None:
                progress_cb(uploaded, total, failed)

    logger.info(
        "Roboflow export finished: %d uploaded, %d failed (batch=%r)", uploaded, failed, batch_name
    )
    return uploaded, failed, failures
