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
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.slugify import slugify
from app.models.ml_model import MLModel, ModelKind
from app.services.inference.detector import DetectionModel, ModelLoadError, PoseModel
from app.services.inference.ultralytics_adapter import UltralyticsDetectionModel, UltralyticsPoseModel

_detection_cache: dict[str, DetectionModel] = {}
_pose_cache: dict[str, PoseModel] = {}


def _probe_class_names(weights_path: str, kind: ModelKind) -> dict[int, str]:
    if kind == ModelKind.DETECTOR:
        return UltralyticsDetectionModel(weights_path).class_names
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
    can't be loaded — never registers a model the platform can't run."""
    class_names = _probe_class_names(weights_path, kind)
    class_config = [{"id": class_id, "name": cname} for class_id, cname in sorted(class_names.items())]

    model = MLModel(
        name=name,
        version=version,
        kind=kind,
        framework=framework,
        weights_path=weights_path,
        class_config=class_config,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


def _download_weights(url: str, name: str) -> Path:
    """Stream `url` onto disk under MODELS_DIR/pt, returning the local path.
    Runs before any DB row exists, so a failed/invalid download never
    registers anything — see register_model_from_url."""
    scheme = urlsplit(url).scheme
    if scheme not in ("http", "https"):
        raise ModelLoadError(f"Unsupported URL scheme {scheme!r} — only http/https links are allowed")

    suffix = Path(urlsplit(url).path).suffix or ".pt"
    dest_dir = settings.MODELS_DIR / "pt"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{slugify(name)}-{uuid.uuid4().hex[:8]}{suffix}"

    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as response:
            response.raise_for_status()
            with dest.open("wb") as f:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
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
    _pose_cache.pop(key, None)


def delete_model(db: Session, model: MLModel) -> None:
    evict_model_cache(model.id)

    # Only remove the file if it's a weights file this app manages (under
    # ARTIFACTS_DIR) — never delete something a manually-entered path might
    # point at outside that tree.
    weights_path = Path(model.weights_path)
    try:
        weights_path.relative_to(settings.ARTIFACTS_DIR)
    except ValueError:
        pass
    else:
        weights_path.unlink(missing_ok=True)

    db.delete(model)
    db.commit()


def get_detection_model(db: Session, model_id: uuid.UUID) -> DetectionModel:
    key = str(model_id)
    if key in _detection_cache:
        return _detection_cache[key]

    model = db.get(MLModel, model_id)
    if model is None:
        raise ModelLoadError(f"No model registered with id {model_id}")
    if model.kind != ModelKind.DETECTOR:
        raise ModelLoadError(f"Model {model.name!r} is a {model.kind.value}, not a DETECTOR")

    instance = UltralyticsDetectionModel(model.weights_path)
    _detection_cache[key] = instance
    return instance


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
