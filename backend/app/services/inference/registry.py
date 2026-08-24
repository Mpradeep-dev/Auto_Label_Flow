"""Model registration + a process-level load-once cache.

Registration reads the class list straight from the weights (`model.names`)
— this is the concrete mechanism behind PLAN Decision 1 ("class taxonomy is
read from the model, never hardcoded"). The cache exists so a request
handler or Celery task never pays YOLO's load cost more than once per
process; see PLAN "Jobs: Celery + Redis" for why this matters for the
`gpu` queue specifically (concurrency=1, one process, one cache).
"""
from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

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
