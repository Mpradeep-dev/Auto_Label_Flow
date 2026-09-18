"""Pull an existing Roboflow project version in as a new Dataset (PLAN
follow-on: "import source", the other half of the Roboflow integration).

Imported boxes are treated as ground truth (`source=HUMAN` — someone
labeled them, just not in this app) but images are left `PENDING` so
the user must review and approve them before they can be versioned or
exported, same as auto-annotated images. This ensures every image
passes through the review flow.

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

import logging
import shutil
import tempfile
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable, Iterator, TypeVar

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
from app.services.annotation.service import create_annotation
from app.services.integrations.roboflow_connect import get_client
from app.services.storage.factory import get_storage

logger = logging.getLogger(__name__)

_SPLIT_DIRS = ("train", "valid", "test")
_RAW_SEARCH_PAGE_SIZE = 100

# Each image here costs a network round-trip (raw pull: a details fetch plus
# a CDN download; versioned pull: a storage write) with no dependency on the
# others — pulling them one at a time made a multi-hundred-image import
# spend nearly all its wall-clock time on network latency. `_run_windowed`
# below fetches up to this many concurrently; DB writes (which touch a
# single non-thread-safe `Session`) always happen back on the caller's own
# thread, never inside a worker.
_IMPORT_MAX_WORKERS = 8

# A `db.commit()` is a full transaction sync — real work compared to the
# `db.flush()` that's actually needed mid-loop (just to populate the new
# `Image` row's client-side UUID default before annotations reference it by
# id). Flushing every image but only committing every `_COMMIT_BATCH_SIZE`
# of them (plus once more at the end, unconditionally) cuts commit count
# from O(images) to O(images / batch) without changing what ends up
# persisted — nothing here is rolled back on failure either way, so there's
# no atomicity this batching gives up.
_COMMIT_BATCH_SIZE = 50

T = TypeVar("T")
R = TypeVar("R")


def _run_windowed(
    items: list[T],
    fetch: Callable[[T], R],
    *,
    max_workers: int,
    should_cancel: Callable[[], bool] | None,
) -> Iterator[tuple[T, R | None, Exception | None]]:
    """Runs `fetch(item)` for up to `max_workers` items at a time, yielding
    `(item, result, error)` as each finishes — in completion order, not
    input order, since every caller here only aggregates counts/rows and
    never depends on ordering. `should_cancel()` is checked before topping
    the window back up (not before every yield — there's no way to
    interrupt a fetch already in flight), so a cancel requested mid-run can
    still let up to `max_workers - 1` already-started fetches complete and
    be yielded before the generator stops."""
    if not items:
        return
    workers = max(1, min(max_workers, len(items)))
    pending_iter = iter(items)
    in_flight: dict[Future, T] = {}

    def _cancelled() -> bool:
        return should_cancel is not None and should_cancel()

    with ThreadPoolExecutor(max_workers=workers) as pool:

        def _top_up() -> None:
            while len(in_flight) < workers and not _cancelled():
                try:
                    item = next(pending_iter)
                except StopIteration:
                    return
                future = pool.submit(fetch, item)
                in_flight[future] = item

        _top_up()
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                item = in_flight.pop(future)
                try:
                    yield item, future.result(), None
                except Exception as exc:  # noqa: BLE001 - surfaced to the caller, not swallowed here
                    yield item, None, exc
            if _cancelled():
                for future in in_flight:
                    future.cancel()
                break
            _top_up()

# Roboflow's /search occasionally throws a transient 5xx (observed: bare
# HTTP 500 "contact support" bodies for a few minutes at a time) or a 429.
# Without a retry, one blip fails the whole multi-page import job. Retry
# those statuses and connection/timeout errors with exponential backoff;
# a 4xx or an {"error": ...} envelope is not transient and still raises at
# once. Backoff between the 4 attempts: 1s, 2s, 4s.
_SEARCH_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_SEARCH_MAX_ATTEMPTS = 4
_SEARCH_BACKOFF_BASE_S = 1.0

# `rf_project.image()` (the SDK's per-image detail fetch, used by the raw
# pull below) has the identical weakness `_rf_search_page` was written to
# route around for `/search`: it ends with a bare `requests.get(url).json()`
# — no status check, no retry, and critically no *timeout* — so a transient
# 5xx/429 (empty or HTML body) surfaces only as an opaque
# `json.JSONDecodeError` ("Expecting value: line 2 column 1 (char 1)",
# observed live), and a connection that hangs instead of erroring blocks
# this fetch forever. That second failure mode is worse than it sounds
# under `_run_windowed`: several of these run concurrently, and
# `should_cancel()` there is only re-checked once `wait()` returns a
# completed future — if every in-flight fetch is hung, `wait()` never
# returns, so nothing is left running to ever notice the cancel flag. The
# whole import freezes (progress stalls, Cancel does nothing) instead of
# stopping. Reimplemented raw (same URL/params the SDK builds from
# `rf_project.id`) with an explicit timeout so a hang always surfaces as a
# retryable `requests.RequestException` instead of blocking forever;
# skipping that one image once retries are exhausted (see
# `_rf_image_details`) keeps a single blip from aborting an otherwise-
# healthy multi-hundred-image import.
_IMAGE_DETAIL_MAX_ATTEMPTS = 3
_IMAGE_DETAIL_BACKOFF_BASE_S = 1.0
_IMAGE_DETAIL_TIMEOUT_S = 30

# `Version.download()` (the versioned pull's one big blocking call — wait
# for export generation, then fetch a status/link endpoint, then download
# and extract the zip) raises a bare `RuntimeError` on any non-200/202
# response from that status/link endpoint, including a transient 5xx/429,
# with the original status code already lost by the time it reaches here —
# so unlike the raw pull's retries above, this can't tell transient and
# permanent failures apart by status. Retried anyway, on any exception:
# without it, one blip aborts what's often a multi-minute download outright,
# and a genuine permanent failure (bad version, deleted project) just fails
# a few seconds later than it would have.
_VERSION_DOWNLOAD_MAX_ATTEMPTS = 3
_VERSION_DOWNLOAD_BACKOFF_BASE_S = 2.0


def _describe_unreachable(exc: requests.RequestException, attempts: int) -> str:
    """A `requests.exceptions.ConnectionError` (which is what a DNS
    resolution failure like `NameResolutionError` surfaces as) means the
    request never reached Roboflow at all — that's a local network/DNS/
    firewall/proxy problem on this machine, not "Roboflow is having a bad
    moment", and telling the user to just retry in a few minutes is wrong
    advice for it (observed live: users read that hint, wait, and hit the
    exact same DNS failure again). Anything else (timeouts, etc.) keeps the
    original "transient, retry later" framing, which is accurate for those."""
    if isinstance(exc, requests.exceptions.ConnectionError):
        return (
            f"Could not reach Roboflow (api.roboflow.com) after {attempts} attempts — this "
            "machine was unable to connect at all, which usually means a local internet, DNS, "
            f"or firewall/proxy problem rather than an issue on Roboflow's side. ({exc})"
        )
    return (
        f"Roboflow /search could not be reached after {attempts} attempts ({exc}). This is "
        "usually a temporary Roboflow-side issue — retry the import in a few minutes."
    )


def _download_version_dataset(rf_version, model_format: str, location: str):
    """Retries `rf_version.download(model_format, location=location)` as a
    whole (see the module-level comment on `_VERSION_DOWNLOAD_MAX_ATTEMPTS`
    for why this can't be smarter about which failures are worth retrying).

    `location` is cleared before every attempt: `Version.download()` treats
    an already-existing `location` as "already downloaded" and returns
    immediately without downloading anything (`overwrite` defaults to
    `False`) — without clearing it first, a retry after a failure that left
    partial files behind (e.g. died mid zip-extract) would silently return
    that partial, corrupt dataset instead of actually re-downloading."""
    last_exc: Exception | None = None
    for attempt in range(1, _VERSION_DOWNLOAD_MAX_ATTEMPTS + 1):
        if Path(location).exists():
            shutil.rmtree(location, ignore_errors=True)
        try:
            return rf_version.download(model_format, location=location)
        except Exception as exc:  # noqa: BLE001 — status code already lost by the SDK; see comment above
            last_exc = exc
            if attempt == _VERSION_DOWNLOAD_MAX_ATTEMPTS:
                break
            logger.warning(
                "Roboflow version download failed (attempt %d/%d): %s — retrying",
                attempt,
                _VERSION_DOWNLOAD_MAX_ATTEMPTS,
                exc,
            )
            time.sleep(_VERSION_DOWNLOAD_BACKOFF_BASE_S * 2 ** (attempt - 1))
    assert last_exc is not None
    raise last_exc


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
    images_only: bool = False,
    progress_cb: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Dataset:
    """`images_only` still downloads and persists every image but skips
    turning its YOLO label file into annotations, so it lands with zero
    boxes for auto-annotate to start fresh on (issue #22: existing
    Roboflow labels get in the way of using auto-annotate on pulled-in
    images).

    `progress_cb(current, total)`, if given, is called once with
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
        rf_dataset = _download_version_dataset(rf_version, "yolov8", str(Path(tmp) / "download"))
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

        def _stage_image(item: tuple[str, Path]) -> tuple[str, int, int] | None:
            """Runs off the main thread: local decode (to get dimensions)
            plus the storage write, neither of which touches `db`. Returns
            `None` for an unreadable image, same as the old inline `continue`."""
            _split, image_path = item
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            height, width = img.shape[:2]
            key = safe_storage_key(
                str(project_id), str(dataset.id), "images", original_filename=image_path.name
            )
            storage.upload(image_path, key, content_type="image/jpeg")
            return key, width, height

        processed = 0
        for (split, image_path), result, exc in _run_windowed(
            pending, _stage_image, max_workers=_IMPORT_MAX_WORKERS, should_cancel=should_cancel
        ):
            processed += 1
            if exc is not None:
                logger.warning("Roboflow import: failed to stage %s: %s", image_path.name, exc, exc_info=True)
            elif result is not None:
                key, width, height = result
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
                db.flush()

                labels_dir = location / split / "labels"
                label_path = labels_dir / f"{image_path.stem}.txt"
                if not images_only and label_path.exists():
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
                if processed % _COMMIT_BATCH_SIZE == 0:
                    db.commit()

            if progress_cb is not None:
                progress_cb(processed, total)

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


def _rf_search_page(
    rf_project,
    api_key: str,
    *,
    offset: int,
    limit: int,
    fields: list[str],
    batch_id: str | None = None,
) -> list[dict]:
    """Direct call to Roboflow's `/search` endpoint, in place of
    `rf_project.search()`. The SDK (roboflow==1.4.1) ends `search()` with a
    bare `data.json()["results"]` — no status check, no error-envelope
    check (unlike `Workspace.create_project()` right beside it, which does
    `if "error" in r.json()`) — so any response that isn't
    `{"results": [...]}` (an `{"error": ...}` body, a changed response
    shape, a permission failure on this one endpoint) surfaces only as an
    opaque `KeyError: 'results'` from deep in the SDK with the real cause
    swallowed. This issues the identical request (same URL and payload
    shape the SDK builds from `rf_project.id`) but inspects the response,
    retries a transient 5xx/429, and keeps the body in both the raised
    error and the logs.

    `batch_id`, when given, narrows results to that one upload batch —
    same as the SDK's `search(batch=True, batch_id=...)`.
    """
    from roboflow.config import API_URL

    url = f"{API_URL}/{rf_project.id}/search?api_key={api_key}"
    payload = {
        "offset": offset,
        "limit": limit,
        "batch": batch_id is not None,
        "fields": fields,
    }
    if batch_id is not None:
        payload["batch_id"] = batch_id

    for attempt in range(1, _SEARCH_MAX_ATTEMPTS + 1):
        try:
            resp = requests.post(url, json=payload, timeout=30)
        except requests.RequestException as exc:
            logger.warning(
                "Roboflow /search request error (attempt %d/%d): %s",
                attempt,
                _SEARCH_MAX_ATTEMPTS,
                exc,
            )
            if attempt == _SEARCH_MAX_ATTEMPTS:
                raise RuntimeError(_describe_unreachable(exc, _SEARCH_MAX_ATTEMPTS)) from exc
            time.sleep(_SEARCH_BACKOFF_BASE_S * 2 ** (attempt - 1))
            continue

        if resp.status_code in _SEARCH_RETRY_STATUSES and attempt < _SEARCH_MAX_ATTEMPTS:
            logger.warning(
                "Roboflow /search returned HTTP %s (attempt %d/%d) — retrying",
                resp.status_code,
                attempt,
                _SEARCH_MAX_ATTEMPTS,
            )
            time.sleep(_SEARCH_BACKOFF_BASE_S * 2 ** (attempt - 1))
            continue

        break

    try:
        body = resp.json()
    except ValueError as exc:
        logger.error(
            "Roboflow /search returned non-JSON (HTTP %s): %s", resp.status_code, resp.text[:2000]
        )
        raise RuntimeError(
            f"Roboflow /search returned a non-JSON response (HTTP {resp.status_code})"
        ) from exc

    if resp.status_code != 200 or not isinstance(body, dict) or "results" not in body:
        logger.error("Roboflow /search failed (HTTP %s): %s", resp.status_code, body)
        detail = (body.get("error") or body.get("message") or body) if isinstance(body, dict) else body
        hint = (
            " This is usually a temporary Roboflow-side issue — retry the import in a few minutes."
            if resp.status_code in _SEARCH_RETRY_STATUSES
            else ""
        )
        raise RuntimeError(f"Roboflow /search failed (HTTP {resp.status_code}): {detail}{hint}")

    return body["results"]


def _rf_image_details(rf_project, api_key: str, image_id: str) -> dict | None:
    """Direct call to Roboflow's per-image detail endpoint, in place of
    `rf_project.image(image_id)` — see the module-level comment on
    `_IMAGE_DETAIL_MAX_ATTEMPTS` for why the SDK call's missing timeout
    needed routing around, not just retrying. Issues the identical request
    (same URL the SDK builds from `rf_project.id`) but bounds it with
    `timeout=_IMAGE_DETAIL_TIMEOUT_S`, and keeps the SDK's own `{"error":
    ...}` / missing-"image" checks so those still retry exactly as before.
    Returns `None` once attempts are exhausted, so the caller can skip just
    this one image — same as an unreadable image file a few lines below —
    instead of the whole job dying or hanging over one blip.

    `api_key` is sent as a query param (`params=`, not baked into `url`)
    and every log line below logs only `type(exc).__name__`, never `exc`
    itself or `url` — a `requests`/urllib3 connection-level exception's own
    message routinely echoes the full request URL it tried, key included,
    and that string reaching a log line is exactly how a key leaks into
    log aggregation."""
    from roboflow.config import API_URL

    url = f"{API_URL}/{rf_project.id}/images/{image_id}"

    for attempt in range(1, _IMAGE_DETAIL_MAX_ATTEMPTS + 1):
        try:
            resp = requests.get(url, params={"api_key": api_key}, timeout=_IMAGE_DETAIL_TIMEOUT_S)
            body = resp.json()
            if "error" in body:
                raise RuntimeError(body["error"])
            if "image" not in body:
                raise RuntimeError("Image not found")
            return body["image"]
        except (ValueError, RuntimeError, requests.RequestException) as exc:
            if attempt == _IMAGE_DETAIL_MAX_ATTEMPTS:
                logger.warning(
                    "Roboflow image detail fetch failed for %s after %d attempts: %s",
                    image_id,
                    attempt,
                    type(exc).__name__,
                )
                return None
            logger.warning(
                "Roboflow image detail fetch failed for %s (attempt %d/%d): %s — retrying",
                image_id,
                attempt,
                _IMAGE_DETAIL_MAX_ATTEMPTS,
                type(exc).__name__,
            )
            time.sleep(_IMAGE_DETAIL_BACKOFF_BASE_S * 2 ** (attempt - 1))
    return None


def import_roboflow_raw_project(
    db: Session,
    *,
    project_id: uuid.UUID,
    workspace: str,
    project_slug: str,
    dataset_name: str | None,
    unannotated_only: bool = False,
    batch_id: str | None = None,
    images_only: bool = False,
    progress_cb: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Dataset:
    """`images_only` still downloads and persists every pulled image but
    skips turning its `annotation.boxes` into annotations, so it lands with
    zero boxes for auto-annotate to start fresh on (issue #22) — orthogonal
    to `unannotated_only`, which instead narrows *which* images get pulled.

    Pulls a project's raw uploaded images directly, for a project with
    no generated Version to `.download()`. Images are left PENDING so
    the user must review and approve them before they can be versioned
    or exported — same as `import_roboflow_project`.

    `unannotated_only` narrows the pull to images with zero existing
    Roboflow annotations — useful when the point of pulling them in here
    is specifically to label what nobody's touched yet, rather than
    re-reviewing images Roboflow already has boxes on. Filtered locally
    off each item's `annotations.count` (the `search()` endpoint has no
    server-side "has no annotations" filter to push this down to).

    `batch_id`, when given, narrows the pull to one upload batch (as
    listed by `roboflow_browse.list_batches`) instead of every raw image
    in the project — pushed down to the `/search` request itself rather
    than filtered locally, since Roboflow does support it server-side.

    `should_cancel()` is checked both between search pages (this path has
    no single giant blocking call the way a version `.download()` does, so
    even the listing phase can stop early) and before each image."""
    project = db.get(Project, project_id)
    if project is None:
        raise ValueError(f"No project with id {project_id}")

    rf, config = get_client(db)
    rf_project = rf.workspace(workspace).project(project_slug)

    # "annotations" is only asked for when `unannotated_only` actually needs
    # it (to filter locally on `annotations.count`) — on a project with a
    # lot of boxes per image, serializing that field for every result is
    # real extra work on Roboflow's side for no reason the other branch
    # cares about, so it's left off the payload entirely there.
    search_fields = ["id", "name"] + (["annotations"] if unannotated_only else [])

    items: list[dict] = []
    offset = 0
    while True:
        if should_cancel is not None and should_cancel():
            break
        page = _rf_search_page(
            rf_project,
            config["api_key"],
            offset=offset,
            limit=_RAW_SEARCH_PAGE_SIZE,
            fields=search_fields,
            batch_id=batch_id,
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

    def _stage_raw_image(item: dict) -> dict | None:
        """Runs off the main thread: the details fetch, the CDN download,
        the decode, and the storage write — none of it touches `db`.
        Returns `None` for any of the several "nothing to import for this
        item" cases the old inline `continue`s handled (missing details,
        missing URL, failed download, undecodable image), same as before."""
        details = _rf_image_details(rf_project, config["api_key"], item["id"])
        if details is None:
            return None
        url = (details.get("urls") or {}).get("original")
        if not url:
            return None
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Roboflow image download failed for %s: %s", item["id"], exc)
            return None
        arr = cv2.imdecode(np.frombuffer(resp.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            return None
        height, width = arr.shape[:2]
        original_filename = details.get("name") or f"{item['id']}.jpg"
        key = safe_storage_key(str(project_id), str(dataset.id), "images", original_filename=original_filename)
        storage.upload_bytes(resp.content, key, content_type="image/jpeg")
        return {
            "key": key,
            "original_filename": original_filename,
            "width": width,
            "height": height,
            "annotation": details.get("annotation") or {},
        }

    processed = 0
    for item, result, exc in _run_windowed(
        items, _stage_raw_image, max_workers=_IMPORT_MAX_WORKERS, should_cancel=should_cancel
    ):
        processed += 1
        if exc is not None:
            logger.warning("Roboflow import: failed to stage %s: %s", item.get("id"), exc, exc_info=True)
        elif result is not None:
            image = Image(
                project_id=project_id,
                dataset_id=dataset.id,
                storage_key=result["key"],
                original_filename=result["original_filename"],
                width=result["width"],
                height=result["height"],
                source_type=ImageSourceType.UPLOAD,
            )
            db.add(image)
            db.flush()

            annotation = result["annotation"]
            ann_w = annotation.get("width") or result["width"]
            ann_h = annotation.get("height") or result["height"]
            for box in ([] if images_only else annotation.get("boxes") or []):
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
            if processed % _COMMIT_BATCH_SIZE == 0:
                db.commit()

        if progress_cb is not None:
            progress_cb(processed, total)

    db.commit()
    return dataset
