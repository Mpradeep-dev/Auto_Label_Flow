"""Model registration + a process-level load-once cache.

Registration reads the class list straight from the weights (`model.names`)
— this is the concrete mechanism behind PLAN Decision 1 ("class taxonomy is
read from the model, never hardcoded"). The cache exists so a request
handler or Celery task never pays YOLO's load cost more than once per
process; see PLAN "Jobs: Celery + Redis" for why this matters for the
`gpu` queue specifically (concurrency=1, one process, one cache).
"""
from __future__ import annotations

import shutil
import threading
import uuid
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import UnsafeUrlError, assert_public_host
from app.core.slugify import slugify
from app.models.ml_model import MLModel, ModelKind
from app.models.project import Project
from app.services.inference.detector import DetectionModel, ModelLoadError, PoseModel
from app.services.inference.ultralytics_adapter import UltralyticsDetectionModel, UltralyticsPoseModel
from app.services.inference.yolo_world_adapter import YoloWorldDetectionModel

_YOLO_WORLD_FRAMEWORK = "yolo-world"

_detection_cache: dict[str, DetectionModel] = {}
_detection_locks: dict[str, threading.Lock] = {}
_pose_cache: dict[str, PoseModel] = {}


def _detection_model_cls(framework: str) -> type[UltralyticsDetectionModel]:
    return YoloWorldDetectionModel if framework == _YOLO_WORLD_FRAMEWORK else UltralyticsDetectionModel


def _probe_class_names(weights_path: str, kind: ModelKind, framework: str = "ultralytics") -> dict[int, str]:
    if kind == ModelKind.DETECTOR:
        return _detection_model_cls(framework)(weights_path).class_names
    from ultralytics import YOLO

    try:
        model = YOLO(weights_path)
    except Exception as exc:
        raise ModelLoadError(f"Failed to load weights at {weights_path}: {exc}") from exc
    return {int(k): v for k, v in model.names.items()}


def register_model(
    db: Session,
    *,
    name: str,
    weights_path: str,
    kind: ModelKind,
    version: str = "v1",
    framework: str = "ultralytics",
) -> MLModel:
    """Load `weights_path` once to read its class map, then persist a
    `MLModel` row. Raises ModelLoadError (surfaced as 400) if the weights
    can't be loaded — never registers a model the platform can't run.

    For an open-vocabulary framework (`framework="yolo-world"`), the probed
    class map is just the checkpoint's default vocabulary — stored as a
    display fallback in `class_config`, but the row is marked
    `is_promptable=True` so callers know to supply real classes at
    inference time instead of trusting it."""
    class_names = _probe_class_names(weights_path, kind, framework)
    class_config = [{"id": class_id, "name": cname} for class_id, cname in sorted(class_names.items())]

    model = MLModel(
        name=name,
        version=version,
        kind=kind,
        framework=framework,
        weights_path=weights_path,
        class_config=class_config,
        is_promptable=(framework == _YOLO_WORLD_FRAMEWORK),
    )
    db.add(model)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise ModelLoadError(
            f"A model named {name!r} with version {version!r} is already registered — "
            "use a different name/version, or delete the existing one first."
        ) from exc
    db.refresh(model)
    return model


_MAX_REDIRECTS = 5


def _assert_public_host(url: str) -> None:
    """Runs on the initial URL *and* on every redirect hop in
    `_download_weights` (SEC-02) — a public-looking URL that redirects to an
    internal address is exactly as dangerous as one that points there
    directly, so both must be checked, not just the first. Thin wrapper over
    the shared `core.security` guard: translates its generic
    `UnsafeUrlError` into this module's `ModelLoadError`, so callers of
    `register_model_from_url` see the exception type they already do."""
    try:
        assert_public_host(url)
    except UnsafeUrlError as exc:
        raise ModelLoadError(str(exc)) from exc


def _download_weights(url: str, name: str) -> Path:
    """Stream `url` onto disk under MODELS_DIR/pt, returning the local path.
    Runs before any DB row exists, so a failed/invalid download never
    registers anything — see register_model_from_url.

    Manually follows redirects (rather than httpx's `follow_redirects=True`)
    so every hop — not just the URL the caller supplied — is re-checked by
    `_assert_public_host` before being fetched."""
    scheme = urlsplit(url).scheme
    if scheme not in ("http", "https"):
        raise ModelLoadError(f"Unsupported URL scheme {scheme!r} — only http/https links are allowed")
    _assert_public_host(url)

    suffix = Path(urlsplit(url).path).suffix or ".pt"
    dest_dir = settings.MODELS_DIR / "pt"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{slugify(name)}-{uuid.uuid4().hex[:8]}{suffix}"

    try:
        current_url = url
        with httpx.Client(follow_redirects=False, timeout=300.0) as client:
            for _ in range(_MAX_REDIRECTS + 1):
                with client.stream("GET", current_url) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            response.raise_for_status()
                            break
                        current_url = urljoin(current_url, location)
                        if urlsplit(current_url).scheme not in ("http", "https"):
                            raise ModelLoadError(f"Redirect to unsupported scheme: {current_url}")
                        _assert_public_host(current_url)
                        continue
                    response.raise_for_status()
                    with dest.open("wb") as f:
                        for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                            f.write(chunk)
                    break
            else:
                raise ModelLoadError(f"Too many redirects fetching {url}")
    except httpx.HTTPError as exc:
        dest.unlink(missing_ok=True)
        raise ModelLoadError(f"Failed to download weights from {url}: {exc}") from exc

    return dest


def register_model_from_url(
    db: Session,
    *,
    name: str,
    url: str,
    kind: ModelKind,
    version: str = "v1",
    framework: str = "ultralytics",
) -> MLModel:
    """Downloads `url` into ARTIFACTS_DIR then registers it exactly like
    register_model. Deletes the downloaded file if it turns out not to be
    loadable weights, so a bad link never leaves an orphaned file behind."""
    weights_path = _download_weights(url, name)
    try:
        return register_model(db, name=name, weights_path=str(weights_path), kind=kind, version=version, framework=framework)
    except ModelLoadError:
        weights_path.unlink(missing_ok=True)
        raise


def _place_weights_file(temp_path: Path, name: str, suffix: str) -> Path:
    """Move an already-streamed-to-disk temp file (see
    core/security.stream_upload_to_temp) into MODELS_DIR/pt under a
    collision-proof name."""
    dest_dir = settings.MODELS_DIR / "pt"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{slugify(name)}-{uuid.uuid4().hex[:8]}{suffix}"
    shutil.move(str(temp_path), dest)
    return dest


def register_model_from_upload(
    db: Session,
    *,
    name: str,
    temp_path: Path,
    suffix: str,
    kind: ModelKind,
    version: str = "v1",
    framework: str = "ultralytics",
) -> MLModel:
    """Browser-upload counterpart to register_model_from_url: places a
    weights file the user picked from their own machine into ARTIFACTS_DIR,
    then registers it exactly like register_model."""
    weights_path = _place_weights_file(temp_path, name, suffix)
    try:
        return register_model(db, name=name, weights_path=str(weights_path), kind=kind, version=version, framework=framework)
    except ModelLoadError:
        weights_path.unlink(missing_ok=True)
        raise


def rename_model(db: Session, model: MLModel, name: str) -> MLModel:
    model.name = name
    db.commit()
    db.refresh(model)
    return model


def evict_model_cache(model_id: uuid.UUID) -> None:
    key = str(model_id)
    _detection_cache.pop(key, None)
    _detection_locks.pop(key, None)
    _pose_cache.pop(key, None)


class ModelInUseError(RuntimeError):
    """Raised when a model can't be deleted because something still
    references it (audit finding DB-01) — e.g. an InferenceJob, whose FK to
    `models.id` is RESTRICT by design (a job's history shouldn't silently
    lose which model produced it). The API layer turns this into a 409."""


def delete_model(db: Session, model: MLModel) -> None:
    evict_model_cache(model.id)

    db.delete(model)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise ModelInUseError(
            f"Model {model.name!r} can't be deleted — it's still referenced by existing inference or "
            "training job history. Delete or reassign those first."
        ) from exc

    # Only remove the file if it's a weights file this app manages (under
    # ARTIFACTS_DIR) — never delete something a manually-entered path might
    # point at outside that tree. Runs only after the DB delete actually
    # succeeds, so a blocked delete never leaves the model row intact but
    # its weights file gone.
    weights_path = Path(model.weights_path)
    try:
        weights_path.relative_to(settings.ARTIFACTS_DIR)
    except ValueError:
        return
    weights_path.unlink(missing_ok=True)


def get_detection_model(
    db: Session, model_id: uuid.UUID, class_names: list[str] | None = None
) -> DetectionModel:
    """Load (or reuse the cached) detector for `model_id`.

    `class_names` matters only for a promptable (YOLO-World) model: it's
    the prompt list — typically a project's `class_config` names — that
    gets re-embedded via `set_classes()` before the instance is handed
    back, so every caller (Celery batch job, sync predict/auto-annotate
    endpoint) always gets a detector pointed at the right vocabulary. This
    mutates the shared, process-cached instance in place, so it runs under
    a per-model lock: the `gpu` Celery queue is already serialized
    (concurrency=1), but the synchronous FastAPI endpoints are not."""
    key = str(model_id)
    model = db.get(MLModel, model_id)
    if model is None:
        raise ModelLoadError(f"No model registered with id {model_id}")
    if model.kind != ModelKind.DETECTOR:
        raise ModelLoadError(f"Model {model.name!r} is a {model.kind.value}, not a DETECTOR")

    lock = _detection_locks.setdefault(key, threading.Lock())
    with lock:
        instance = _detection_cache.get(key)
        if instance is None:
            instance = _detection_model_cls(model.framework)(model.weights_path)
            _detection_cache[key] = instance

        if model.is_promptable:
            if not class_names:
                raise ModelLoadError(
                    f"Model {model.name!r} is open-vocabulary — the project needs at least one "
                    "class defined before it can be used for auto-annotation."
                )
            instance.set_classes(class_names)

    return instance


def get_project_class_names(db: Session, project_id: uuid.UUID) -> list[str]:
    """The prompt list for a promptable model — a project's active
    taxonomy, in `class_config` order. Callers pass this straight into
    `get_detection_model(..., class_names=...)`."""
    project = db.get(Project, project_id)
    if project is None:
        raise ModelLoadError(f"No project with id {project_id}")
    return [entry["name"] for entry in project.class_config]


def get_pose_model(db: Session, model_id: uuid.UUID) -> PoseModel:
    key = str(model_id)
    if key in _pose_cache:
        return _pose_cache[key]

    model = db.get(MLModel, model_id)
    if model is None:
        raise ModelLoadError(f"No model registered with id {model_id}")
    if model.kind != ModelKind.POSE:
        raise ModelLoadError(f"Model {model.name!r} is a {model.kind.value}, not a POSE model")

    instance = UltralyticsPoseModel(model.weights_path)
    _pose_cache[key] = instance
    return instance


def clear_cache() -> None:
    """Test-only: drop cached model instances between tests that register
    fresh weights under the same id space."""
    _detection_cache.clear()
    _pose_cache.clear()
