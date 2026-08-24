"""Registry/adapter wiring tests, using a fake detector so they stay fast
and offline — no real weights loaded (house convention, mirrors the sibling
repo's FakeModel-based test fixtures). The real-weights path is covered
separately in test_real_model_integration.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sqlalchemy.orm import Session

from app.models.ml_model import MLModel, ModelKind
from app.services.inference import registry
from app.services.inference.detector import Detection, ModelLoadError


class _FakeDetectionModel:
    """Stands in for UltralyticsDetectionModel — same interface, no ultralytics import."""

    instances_created = 0

    def __init__(self, weights_path: str) -> None:
        _FakeDetectionModel.instances_created += 1
        self._weights_path = weights_path

    @property
    def class_names(self) -> dict[int, str]:
        return {0: "ball", 1: "cone", 2: "cone_1"}

    def predict(self, image, conf=0.20, iou=0.70, imgsz=640) -> list[Detection]:
        return [Detection(class_id=1, class_name="cone", confidence=0.66, x1=0.1, y1=0.1, x2=0.2, y2=0.2)]


@pytest.fixture(autouse=True)
def _reset_cache_and_patch(monkeypatch, tmp_path: Path):
    registry.clear_cache()
    _FakeDetectionModel.instances_created = 0
    monkeypatch.setattr(registry, "UltralyticsDetectionModel", _FakeDetectionModel)
    weights = tmp_path / "fake_detect.pt"
    weights.write_bytes(b"not real weights, never opened by the fake")
    yield weights
    registry.clear_cache()


def test_register_model_reads_class_config_from_weights(db_session: Session, tmp_path: Path) -> None:
    weights = tmp_path / "fake_detect.pt"
    model = registry.register_model(
        db_session, name="detect_v1", weights_path=str(weights), kind=ModelKind.DETECTOR
    )
    assert model.class_config == [
        {"id": 0, "name": "ball"},
        {"id": 1, "name": "cone"},
        {"id": 2, "name": "cone_1"},
    ]
    assert model.kind == ModelKind.DETECTOR


def test_get_detection_model_is_cached_across_calls(db_session: Session, tmp_path: Path) -> None:
    weights = tmp_path / "fake_detect.pt"
    model = registry.register_model(
        db_session, name="detect_v1", weights_path=str(weights), kind=ModelKind.DETECTOR
    )
    # Registration itself probes the weights once to read class_names —
    # reset the counter here so this assertion is purely about the cache.
    _FakeDetectionModel.instances_created = 0
    registry.get_detection_model(db_session, model.id)
    registry.get_detection_model(db_session, model.id)
    assert _FakeDetectionModel.instances_created == 1  # loaded once, reused — not once per call


def test_get_detection_model_rejects_pose_model(db_session: Session, tmp_path: Path) -> None:
    # Constructed directly (not via register_model) so this test doesn't
    # need real pose weights loadable — it's only exercising the
    # kind-mismatch guard in get_detection_model.
    weights = tmp_path / "fake_pose.pt"
    weights.write_bytes(b"x")
    model = MLModel(name="pose_v1", version="v1", kind=ModelKind.POSE, weights_path=str(weights))
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)

    with pytest.raises(ModelLoadError, match="not a DETECTOR"):
        registry.get_detection_model(db_session, model.id)


def test_get_detection_model_missing_id_raises(db_session: Session) -> None:
    import uuid

    with pytest.raises(ModelLoadError, match="No model registered"):
        registry.get_detection_model(db_session, uuid.uuid4())
