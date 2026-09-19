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
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from enum import Enum
from pathlib import Path
from typing import Callable, NamedTuple

import requests
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.models.dataset_version import DatasetVersion
from app.models.image import Image
from app.models.roboflow_job import RoboflowUploadTarget
from app.services.dataset.export_yolo import ExportError, write_yolo_dataset
from app.services.integrations.roboflow_connect import get_client

logger = logging.getLogger(__name__)


class _PushOutcome(Enum):
    """What happened to one pushed image. `ANNOTATION_UPDATED` covers both
    a Roboflow-detected duplicate whose annotation got written via the
    fallback `Project.upload()` path, and (Task 5) an image routed
    directly through `Project.save_annotation()` because we already knew
    its Roboflow id. `UNCHANGED` is a duplicate (or known-id image) with no
    local annotation to push at all — nothing to do, not a failure."""

    NEW_IMAGE = "new_image_uploaded"
    ANNOTATION_UPDATED = "annotation_updated"
    UNCHANGED = "unchanged"


class PushResult(NamedTuple):
    """Return value of `push_version_to_roboflow`. Replaces the old
    `(uploaded, failed, failures, note)` 4-tuple's `uploaded` (which
    conflated genuinely new images with Roboflow-detected duplicates) —
    "duplicate" is expected, not a failure, for this app's own
    import-then-annotate-then-push workflow, so it must never read as
    "nothing happened" when annotations still landed."""

    new_images: int
    annotations_updated: int
    unchanged: int
    failed: int
    failures: list[str]
    note: str | None


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

# Each upload is one blocking HTTP round-trip with no relationship to the
# others — running them one at a time made total export time scale linearly
# with image count for no reason (a multi-hundred-image push spent almost
# all its wall-clock time waiting on network latency, not CPU). Uploads run
# `_EXPORT_MAX_WORKERS` at a time instead; kept modest so a single export
# doesn't hammer Roboflow's upload endpoint hard enough to make its own 429s
# more likely. Fail-fast (below) can now overshoot `_FAIL_FAST_AFTER` by up
# to `_EXPORT_MAX_WORKERS - 1` real attempts, since that many can already be
# in flight before the threshold is noticed — an acceptable bound in
# exchange for concurrency, and still nowhere near "every image."
_EXPORT_MAX_WORKERS = 8


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


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _build_export_note(
    *, new_images: int, annotations_updated: int, unchanged: int, review_job_note: str | None
) -> str | None:
    """Human-readable summary surfaced as `job.error` (an informational
    note, never a failure — see `run_roboflow_export`'s comment on that
    field). Always non-`None` once at least one image succeeded, so the UI
    shows what actually happened instead of staying silent on a fully
    "clean" push — replaces the old behavior where a successful push with
    no review-job problems reported nothing at all. `review_job_note` is
    whatever `_assign_annotating_review_job` returned for the new-image
    batch (its own failure note, or `None`)."""
    succeeded = new_images + annotations_updated + unchanged
    if succeeded == 0:
        return review_job_note

    parts = []
    if new_images:
        parts.append(f"{_plural(new_images, 'new image')} uploaded")
    if annotations_updated:
        parts.append(f"{_plural(annotations_updated, 'existing image')} updated")
    if unchanged:
        parts.append(f"{_plural(unchanged, 'existing image')} unchanged (no local annotation to push)")
    summary = ", ".join(parts) + "."

    if new_images == 0:
        summary += " No new batch or review job was created — nothing new landed on Roboflow."
    elif review_job_note:
        summary += f" {review_job_note}"
    return summary


def _assign_annotating_review_job(project, *, batch_name: str, labeler_email: str | None) -> str | None:
    """Uploading with `is_prediction=True` only gets a pushed image as far
    as Roboflow's "Unassigned" column — moving it into "Annotating" is a
    second, separate API call (`Project.create_annotation_job`) that
    requires an actual labeler + reviewer (a workspace member's email), not
    something upload-time flags alone can do (confirmed against Roboflow's
    own docs: predictions "stay in its batch and stay unassigned" until a
    job is created for them). This finds the batch this export just created
    (by the same name passed to the upload) and files a job for it, using
    the one configured account email as both labeler and reviewer — this
    app has no UI for picking a different person to review its own
    auto-generated predictions.

    Returns `None` on success (or when there's nothing configured to do —
    silently skipping isn't a failure), or a short message when the job
    couldn't be created, for the caller to surface non-fatally: the images
    themselves already uploaded fine either way."""
    if not labeler_email:
        return (
            "Images uploaded, but no default labeler/reviewer email is set (Settings -> Roboflow) — "
            "they're sitting in Roboflow's Unassigned column instead of Annotating. Set that email and "
            "re-export, or create the review job yourself in Roboflow's Annotate tab."
        )

    try:
        batches = project.get_batches().get("batches", [])
        batch = next((b for b in batches if b.get("name") == batch_name), None)
        if batch is None:
            return (
                f"Images uploaded, but couldn't find batch {batch_name!r} on Roboflow afterward to file a "
                "review job for it — they're sitting in Unassigned. Create the job yourself in Roboflow's "
                "Annotate tab."
            )
        project.create_annotation_job(
            name=batch_name,
            batch_id=batch["id"],
            labeler_email=labeler_email,
            reviewer_email=labeler_email,
        )
    except Exception as exc:  # noqa: BLE001 — non-fatal: images already uploaded
        logger.warning("Roboflow export: could not auto-create annotation job for batch %r: %s", batch_name, exc)
        return (
            f"Images uploaded, but auto-creating the Roboflow review job failed ({exc}) — they're sitting "
            f"in Unassigned. Check {labeler_email!r} is a member of this Roboflow workspace, or create the "
            "job yourself in Roboflow's Annotate tab."
        )
    return None


def _is_duplicate_upload(result) -> bool:
    """Roboflow dedupes uploads by exact file content within a project: an
    image whose bytes already exist there doesn't raise — `project.upload()`
    returns normally with that image's `"image"` dict carrying
    `"duplicate": true` instead of landing in the requested batch (see
    `rfapi.upload_image`'s `if not (responsejson.get("success") or
    responsejson.get("duplicate")): raise`). `project.upload()` always
    returns a list here (one entry per call, since a single `image_path` is
    always passed) — defensive about the shape regardless, since it's an
    untyped SDK return and test doubles may stub it more loosely."""
    if not result:
        return False
    try:
        image = result[0].get("image") or {}
    except (AttributeError, TypeError, IndexError):
        return False
    return bool(image.get("duplicate"))


def _upload_one_image(project, *, has_annotation: bool, **upload_kwargs) -> _PushOutcome:
    """`project.upload(**upload_kwargs)` with a bounded retry on a transient
    5xx/429 (classified via `status_code`) or a bare connection/timeout
    error (no `status_code` at all — `getattr(exc, "status_code", None)`
    used to fall through to `None`, which isn't in `_UPLOAD_RETRY_STATUSES`,
    so a plain network blip was never retried and immediately failed the
    image). Re-raises the last error once attempts are exhausted, and
    immediately for any non-transient HTTP status.

    Returns `NEW_IMAGE` for a genuinely new upload, `ANNOTATION_UPDATED`
    when Roboflow reports a pre-existing duplicate AND there was a local
    annotation to push (the SDK's own `single_upload` already resolves the
    duplicate's image id and calls `save_annotation` for it — see
    `_upload_kwargs`'s `annotation_overwrite=True`, without which that
    write silently no-ops on Roboflow's 409), or `UNCHANGED` for a
    duplicate with nothing local to push at all."""
    for attempt in range(1, _UPLOAD_MAX_ATTEMPTS + 1):
        try:
            result = project.upload(**upload_kwargs)
            if _is_duplicate_upload(result):
                if not has_annotation:
                    return _PushOutcome.UNCHANGED
                # `result[0]["annotation"]` is `rfapi.save_annotation`'s own
                # return value — either the real success payload or
                # `{"warn": "already annotated"}` if Roboflow's 409 response
                # slipped past `annotation_overwrite=True` (see
                # `_save_annotation_only`'s docstring for why that would
                # happen). A `"warn"` means nothing actually changed on
                # Roboflow's side, so don't report it as an update.
                annotation = result[0].get("annotation")
                if isinstance(annotation, dict) and "warn" in annotation:
                    return _PushOutcome.UNCHANGED
                return _PushOutcome.ANNOTATION_UPDATED
            return _PushOutcome.NEW_IMAGE
        except Exception as exc:  # noqa: BLE001 - re-raised below, classified by status
            status = getattr(exc, "status_code", None)
            transient = status in _UPLOAD_RETRY_STATUSES or (
                status is None and isinstance(exc, requests.RequestException)
            )
            if not transient or attempt == _UPLOAD_MAX_ATTEMPTS:
                raise
            logger.warning(
                "Roboflow export: %s failed %s (attempt %d/%d) — retrying",
                upload_kwargs.get("image_path"),
                f"HTTP {status}" if status is not None else repr(exc),
                attempt,
                _UPLOAD_MAX_ATTEMPTS,
            )
            time.sleep(_UPLOAD_BACKOFF_BASE_S * 2 ** (attempt - 1))


def _save_annotation_only(
    project,
    *,
    roboflow_image_id: str,
    annotation_path: str,
    annotation_labelmap_path: str,
    batch_name: str,
    is_prediction: bool,
) -> _PushOutcome:
    """Updates the annotation on an already-known Roboflow image directly,
    via `Project.save_annotation()` — never `Project.upload()` — so the
    image's (unchanged) bytes are never re-sent and Roboflow's
    duplicate-detection is never involved at all. `annotation_overwrite`
    is always `True`: the whole point of this path is replacing whatever
    annotation the image already carries (the ground truth pulled in at
    import time, or a prediction from an earlier push) with the current
    local one — the default `overwrite=False` would instead hit Roboflow's
    409 "already annotated" response and silently do nothing (see
    `rfapi.save_annotation`, `roboflow/adapters/rfapi.py:815-821`).

    `annotation_labelmap` must be the loaded `{index: name}` mapping, not
    a bare yaml path — `Project.upload()`'s own internal call chain
    (`single_upload`, `roboflow/core/project.py:604-605`) converts it via
    `load_labelmap()` before ever calling `save_annotation`; calling
    `Project.save_annotation()` directly, as this does, means doing that
    conversion ourselves or Roboflow receives the literal path string as
    the labelmap and silently mislabels every class.

    Retries a transient 5xx/429 or bare network error the same as
    `_upload_one_image`; anything else (in particular a 404 — the stored
    id no longer resolves to an image on Roboflow) is raised for the
    caller (`_push_image`) to fall back to a normal upload for."""
    from roboflow.util.image_utils import load_labelmap

    labelmap = load_labelmap(annotation_labelmap_path)
    for attempt in range(1, _UPLOAD_MAX_ATTEMPTS + 1):
        try:
            annotation, _upload_time, _retry_attempts = project.save_annotation(
                annotation_path=annotation_path,
                annotation_labelmap=labelmap,
                image_id=roboflow_image_id,
                job_name=batch_name,
                is_prediction=is_prediction,
                annotation_overwrite=True,
            )
            # `annotation` is `rfapi.save_annotation`'s return value directly
            # — either the real success payload or `{"warn": "already
            # annotated"}` if Roboflow's 409 response slipped past
            # `annotation_overwrite=True` (see this function's docstring).
            # A `"warn"` means nothing actually changed on Roboflow's side,
            # so don't report it as an update.
            if isinstance(annotation, dict) and "warn" in annotation:
                return _PushOutcome.UNCHANGED
            return _PushOutcome.ANNOTATION_UPDATED
        except Exception as exc:  # noqa: BLE001 - re-raised below, classified by status
            status = getattr(exc, "status_code", None)
            transient = status in _UPLOAD_RETRY_STATUSES or (
                status is None and isinstance(exc, requests.RequestException)
            )
            if not transient or attempt == _UPLOAD_MAX_ATTEMPTS:
                raise
            logger.warning(
                "Roboflow export: annotation update for Roboflow image %s failed %s "
                "(attempt %d/%d) — retrying",
                roboflow_image_id,
                f"HTTP {status}" if status is not None else repr(exc),
                attempt,
                _UPLOAD_MAX_ATTEMPTS,
            )
            time.sleep(_UPLOAD_BACKOFF_BASE_S * 2 ** (attempt - 1))


def _push_image(
    project,
    *,
    roboflow_image_id: str | None,
    upload_kwargs: dict,
    batch_name: str,
    is_prediction: bool,
    annotation_labelmap_path: str,
) -> _PushOutcome:
    """Routes a single image to whichever Roboflow write path applies. An
    image this app already knows the Roboflow id for (imported from this
    same workspace/project, via `import_roboflow_raw_project` — see
    `Image.roboflow_image_id`) skips `Project.upload()` entirely and only
    ever updates its annotation through `_save_annotation_only` — never
    re-uploading bytes Roboflow already has, never depending on its
    duplicate-detection. Everything else (a genuinely new local image, an
    image with no local annotation to push at all, or a known id that
    404s because the Roboflow image was deleted since import) falls
    through to the normal `_upload_one_image` path."""
    has_annotation = upload_kwargs["annotation_path"] is not None
    if roboflow_image_id is not None:
        if not has_annotation:
            return _PushOutcome.UNCHANGED
        try:
            return _save_annotation_only(
                project,
                roboflow_image_id=roboflow_image_id,
                annotation_path=upload_kwargs["annotation_path"],
                annotation_labelmap_path=annotation_labelmap_path,
                batch_name=batch_name,
                is_prediction=is_prediction,
            )
        except Exception as exc:  # noqa: BLE001 - only a 404 falls through; anything else is a real failure
            if getattr(exc, "status_code", None) != 404:
                raise
            logger.warning(
                "Roboflow export: stored Roboflow image id %s no longer exists on Roboflow "
                "(404) — falling back to a fresh upload for %s",
                roboflow_image_id,
                upload_kwargs.get("image_path"),
            )
    return _upload_one_image(project, has_annotation=has_annotation, **upload_kwargs)


def push_version_to_roboflow(
    db: Session,
    *,
    version_id: uuid.UUID,
    workspace: str,
    project_slug: str,
    custom_batch_name: str | None = None,
    upload_target: str = RoboflowUploadTarget.ANNOTATING.value,
    progress_cb: Callable[[int, int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> PushResult:
    """Returns a `PushResult` — new-image / annotation-updated / unchanged /
    failed counts, the failure messages, and a human-readable summary note.

    `upload_target` picks which column of Roboflow's Annotate board a
    pushed image with a local annotation lands in — `UNANNOTATED` strips
    the annotation before upload (image only, regardless of what's local),
    `ANNOTATING` (the default, unchanged from before this param existed)
    sends it as a prediction that Roboflow queues for review, `DATASET`
    sends it as ground truth that Roboflow auto-confirms straight into the
    Dataset column. An image with no local annotation always lands
    unannotated regardless of target — there is nothing to push as a
    prediction or ground truth for it. `ANNOTATING` also tries to move the
    pushed batch into Roboflow's "Annotating" column via a second API call
    once uploads finish — see `_assign_annotating_review_job` — which needs
    a default labeler email configured (Settings -> Roboflow); its failure
    (or that email being unset) is folded into the returned `PushResult.note`
    but never fails the export, since the images themselves already
    uploaded fine.

    `progress_cb(succeeded, total, failed)`, if given, is called once with
    `succeeded=0` as soon as the image count is known (materialization
    finished), then once per image after its upload attempt. `succeeded`
    counts only images that actually reached Roboflow — a run where every
    image fails never advances it, so the UI shows "0 uploaded, N failed"
    rather than a bar creeping forward on work that didn't land.
    `should_cancel()`, checked before each image, stops the loop early —
    whatever's already uploaded to Roboflow stays uploaded, same as a
    cancel partway through any other batch job here."""
    if should_cancel is not None and should_cancel():
        return PushResult(0, 0, 0, 0, [], None)

    rf, config = get_client(db)
    project = rf.workspace(workspace).project(project_slug)

    # Left unset, the SDK groups every upload under its own hardcoded
    # `DEFAULT_BATCH_NAME` ("Pip Package Upload") — meaningless in
    # Roboflow's UI once more than one project or app pushes into the same
    # Roboflow project. Name the batch after this app and the dataset
    # version it came from by default, so it's identifiable at a glance —
    # or, if the caller gave one, a user-chosen name instead: pushing into a
    # project that already carries annotations from a prior push/import
    # stays distinguishable from that older upload without needing a whole
    # separate Roboflow project.
    version = db.get(DatasetVersion, version_id)
    dataset = db.get(Dataset, version.dataset_id) if version is not None else None
    # `{dataset.name}-v{version_number}` matches the naming already used for
    # this version's own export filenames (export_yolo.py/export_coco.py/
    # export_cvat.py) — same identifier, just also visible in Roboflow now.
    # Slugified before use either way: Roboflow silently drops uploads whose
    # `batch` isn't `^[a-z0-9_-]{1,64}$` (see `_sanitize_batch_name`).
    raw_batch_name = (custom_batch_name or "").strip() or (
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

        def _image_uuid(image_path: Path) -> uuid.UUID | None:
            # Every materialized export file is named after its own
            # `Image.id` (see `version_data.out_image_filename` —
            # `f"{image.id}{ext}"`), so the stem is always a real UUID in
            # production. A test double that fakes `write_yolo_dataset`
            # with synthetic names (e.g. "img0.jpg") is the one case this
            # legitimately fails to parse — treated the same as "no known
            # Roboflow id for this image", which is the correct fallback.
            try:
                return uuid.UUID(image_path.stem)
            except ValueError:
                return None

        pending_uuids = {u for u in (_image_uuid(p) for _split, p in pending) if u is not None}
        known_roboflow_ids: dict[uuid.UUID, str] = {}
        if pending_uuids:
            rows = db.execute(
                select(Image.id, Image.roboflow_image_id).where(
                    Image.id.in_(pending_uuids),
                    Image.roboflow_workspace == workspace,
                    Image.roboflow_project_slug == project_slug,
                    Image.roboflow_image_id.is_not(None),
                )
            ).all()
            known_roboflow_ids = {row.id: row.roboflow_image_id for row in rows}

        def _upload_kwargs(split: str, image_path: Path) -> dict:
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
            has_annotation = (
                label_path.exists()
                and label_path.stat().st_size > 0
                and upload_target != RoboflowUploadTarget.UNANNOTATED.value
            )
            return dict(
                image_path=str(image_path),
                annotation_path=str(label_path) if has_annotation else None,
                annotation_labelmap=str(data_yaml_path),
                split=roboflow_split,
                batch_name=batch_name,
                # Ground truth is auto-confirmed by Roboflow and skips
                # straight to the "Dataset" column of the Annotate board.
                # These labels come from our pipeline, not a human, so the
                # default target (`ANNOTATING`) instead pushes them as a
                # prediction: Roboflow then queues the image for review
                # rather than treating it as already done. `DATASET` opts
                # into the ground-truth behavior explicitly, when the
                # caller wants to skip that review step.
                is_prediction=upload_target != RoboflowUploadTarget.DATASET.value,
                # Without this, Roboflow's 409 "already annotated" response
                # to a duplicate image's annotation write is a silent
                # no-op (see `_upload_one_image`'s docstring) — every
                # re-push of an already-imported, already-annotated image
                # would otherwise never actually update anything on
                # Roboflow. Harmless for a genuinely new image (nothing
                # to overwrite yet).
                annotation_overwrite=True,
            )

        new_images = 0
        annotations_updated = 0
        unchanged = 0
        succeeded = 0
        failed = 0
        failures: list[str] = []
        seen_statuses: list[int | None] = []
        fail_fast_error: RoboflowExportError | None = None

        # Uploads run `_EXPORT_MAX_WORKERS` at a time in a sliding window:
        # keep that many in flight, and top the window back up as each one
        # finishes. `should_cancel`/fail-fast are only checked between
        # completions (there's no way to interrupt an upload already in
        # flight), so both can overshoot by up to `max_workers - 1` real
        # attempts beyond the point they were noticed — bounded, and the
        # price of not doing every upload one at a time.
        max_workers = max(1, min(_EXPORT_MAX_WORKERS, len(pending))) if pending else 1
        pending_iter = iter(pending)
        in_flight: dict[Future, Path] = {}

        def _cancelled() -> bool:
            return should_cancel is not None and should_cancel()

        with ThreadPoolExecutor(max_workers=max_workers) as pool:

            def _top_up() -> None:
                while len(in_flight) < max_workers and not _cancelled():
                    try:
                        split, image_path = next(pending_iter)
                    except StopIteration:
                        return
                    kwargs = _upload_kwargs(split, image_path)
                    future = pool.submit(
                        _push_image,
                        project,
                        roboflow_image_id=known_roboflow_ids.get(_image_uuid(image_path)),
                        upload_kwargs=kwargs,
                        batch_name=batch_name,
                        is_prediction=kwargs["is_prediction"],
                        annotation_labelmap_path=kwargs["annotation_labelmap"],
                    )
                    in_flight[future] = image_path

            _top_up()
            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    image_path = in_flight.pop(future)
                    try:
                        outcome = future.result()
                        succeeded += 1
                        if outcome is _PushOutcome.NEW_IMAGE:
                            new_images += 1
                        elif outcome is _PushOutcome.ANNOTATION_UPDATED:
                            annotations_updated += 1
                        else:
                            unchanged += 1
                    except Exception as exc:  # a single bad image shouldn't abort the whole push
                        failed += 1
                        seen_statuses.append(getattr(exc, "status_code", None))
                        detail = _describe_upload_error(exc)
                        failures.append(f"{image_path.name}: {detail}")
                        logger.warning(
                            "Roboflow export: upload failed for %s (%s)", image_path.name, detail, exc_info=True
                        )
                    # Checkpointed for every attempt, success or failure —
                    # otherwise a fail-fast abort below would skip the
                    # caller's progress_cb for the failures that triggered
                    # it, leaving job.failed_count/processed_items on the DB
                    # row understating how many images were actually
                    # attempted.
                    if progress_cb is not None:
                        progress_cb(succeeded, total, failed)
                    # Systemic failure: nothing has landed and the first N
                    # images all failed. Retrying the rest for hours won't
                    # help — stop with a message that names the likely cause
                    # (`run_roboflow_export` puts it on the job row).
                    if fail_fast_error is None and succeeded == 0 and failed >= _FAIL_FAST_AFTER:
                        fail_fast_error = RoboflowExportError(_fail_fast_message(seen_statuses, failures))
                if fail_fast_error is not None or _cancelled():
                    for future in in_flight:
                        future.cancel()
                    break
                _top_up()

        if fail_fast_error is not None:
            raise fail_fast_error

    logger.info(
        "Roboflow export finished: %d new, %d annotations updated, %d unchanged, %d failed (batch=%r)",
        new_images,
        annotations_updated,
        unchanged,
        failed,
        batch_name,
    )

    review_job_note: str | None = None
    if upload_target == RoboflowUploadTarget.ANNOTATING.value and new_images > 0:
        # Only ever called when something new actually needs moving out of
        # Unassigned — a duplicate/known-id annotation update never joins
        # the batch this looks up, so there'd be nothing to find/move for
        # an all-existing-images push (see `_build_export_note` for how
        # that case is reported instead).
        review_job_note = _assign_annotating_review_job(
            project, batch_name=batch_name, labeler_email=config.get("default_labeler_email")
        )

    note = _build_export_note(
        new_images=new_images,
        annotations_updated=annotations_updated,
        unchanged=unchanged,
        review_job_note=review_job_note,
    )

    return PushResult(new_images, annotations_updated, unchanged, failed, failures, note)
