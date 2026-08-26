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


class _FakeYoloWorldDetectionModel(_FakeDetectionModel):
    """Stands in for YoloWorldDetectionModel — adds the promptable-only
    set_classes() hook, mutating class_names the same way the real adapter
    normalizes Ultralytics' list-shaped model.names back to a dict."""

    set_classes_calls: list[list[str]] = []

    def __init__(self, weights_path: str) -> None:
        super().__init__(weights_path)
        self._class_names = {0: "person", 1: "dog"}  # default/COCO-ish vocabulary at load

    @property
    def class_names(self) -> dict[int, str]:
        return self._class_names

    def set_classes(self, classes: list[str]) -> None:
        _FakeYoloWorldDetectionModel.set_classes_calls.append(classes)
        self._class_names = dict(enumerate(classes))


@pytest.fixture(autouse=True)
def _reset_cache_and_patch(monkeypatch, tmp_path: Path):
    registry.clear_cache()
    _FakeDetectionModel.instances_created = 0
    _FakeYoloWorldDetectionModel.set_classes_calls = []
    monkeypatch.setattr(registry, "UltralyticsDetectionModel", _FakeDetectionModel)
    monkeypatch.setattr(registry, "YoloWorldDetectionModel", _FakeYoloWorldDetectionModel)
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


def test_register_model_rejects_duplicate_name_and_version(db_session: Session, tmp_path: Path) -> None:
    """Regression test for audit finding DB-03: registering the same
    name+version twice used to silently create two rows; it should be a
    clean, actionable error instead."""
    weights = tmp_path / "fake_detect.pt"
    registry.register_model(db_session, name="dup", weights_path=str(weights), kind=ModelKind.DETECTOR, version="v1")

    with pytest.raises(ModelLoadError, match="already registered"):
        registry.register_model(
            db_session, name="dup", weights_path=str(weights), kind=ModelKind.DETECTOR, version="v1"
        )


def test_register_yolo_world_model_is_marked_promptable(db_session: Session, tmp_path: Path) -> None:
    weights = tmp_path / "fake_detect.pt"
    model = registry.register_model(
        db_session, name="yw_v1", weights_path=str(weights), kind=ModelKind.DETECTOR, framework="yolo-world"
    )
    assert model.is_promptable is True
    assert model.framework == "yolo-world"
    # class_config still gets populated (from the checkpoint's default
    # vocabulary) — it's just a display fallback, not authoritative.
    assert model.class_config == [{"id": 0, "name": "person"}, {"id": 1, "name": "dog"}]


def test_get_detection_model_calls_set_classes_for_promptable_model(db_session: Session, tmp_path: Path) -> None:
    weights = tmp_path / "fake_detect.pt"
    model = registry.register_model(
        db_session, name="yw_v1", weights_path=str(weights), kind=ModelKind.DETECTOR, framework="yolo-world"
    )
    detector = registry.get_detection_model(db_session, model.id, class_names=["helmet", "vest"])
    assert _FakeYoloWorldDetectionModel.set_classes_calls == [["helmet", "vest"]]
    assert detector.class_names == {0: "helmet", 1: "vest"}


def test_get_detection_model_promptable_without_classes_raises(db_session: Session, tmp_path: Path) -> None:
    weights = tmp_path / "fake_detect.pt"
    model = registry.register_model(
        db_session, name="yw_v1", weights_path=str(weights), kind=ModelKind.DETECTOR, framework="yolo-world"
    )
    with pytest.raises(ModelLoadError, match="open-vocabulary"):
        registry.get_detection_model(db_session, model.id, class_names=[])


def test_closed_vocab_model_ignores_class_names_arg(db_session: Session, tmp_path: Path) -> None:
    """A non-promptable model never has set_classes() called on it, even if
    a caller happens to pass class_names — it's simply unused."""
    weights = tmp_path / "fake_detect.pt"
    model = registry.register_model(
        db_session, name="detect_v1", weights_path=str(weights), kind=ModelKind.DETECTOR
    )
    detector = registry.get_detection_model(db_session, model.id, class_names=["irrelevant"])
    assert detector.class_names == {0: "ball", 1: "cone", 2: "cone_1"}


def test_deleting_base_model_sets_lineage_null_not_fk_error(db_session: Session, tmp_path: Path) -> None:
    """Regression test for audit finding DB-02: base_model_id used to be a
    bare UUID column with no FK — this confirms it's now a real,
    SET-NULL-on-delete foreign key, matching TrainingJob's lineage columns,
    rather than a value that could silently point at a deleted model."""
    weights = tmp_path / "fake_detect.pt"
    base = registry.register_model(db_session, name="base", weights_path=str(weights), kind=ModelKind.DETECTOR)
    child = registry.register_model(
        db_session, name="child", weights_path=str(weights), kind=ModelKind.DETECTOR, version="v2"
    )
    child.base_model_id = base.id
    db_session.commit()

    registry.delete_model(db_session, base)

    db_session.refresh(child)
    assert child.base_model_id is None
