"""Integration test against the REAL detect_v1.pt weights and a real frame
from the corpus this platform was built to fix — not a fake/offline test by
design. This is the concrete check that the adapter + letterbox port +
registry pipeline reproduces the same numbers measured directly against the
sibling repo's weights while writing the project plan (see PLAN "I
reproduced this live before planning around it").

Skipped automatically if the weights or the source video aren't present on
this machine (e.g. a CI runner without the corpus) — everything else in the
suite stays fast and offline; only this module pays the real-model cost.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from app.core.config import settings
from app.services.inference.ultralytics_adapter import UltralyticsDetectionModel

DETECT_WEIGHTS = settings.MODELS_DIR / "pt" / "detect_v1.pt"
SAMPLE_VIDEO = (
    Path(r"C:\Users\prade\Desktop\GTP\video_ai\New_videos_players\ex22_Mayur.mov")
)

pytestmark = pytest.mark.skipif(
    not DETECT_WEIGHTS.exists() or not SAMPLE_VIDEO.exists(),
    reason="real weights or sample corpus not present on this machine",
)


@pytest.fixture(scope="module")
def frame_100() -> "cv2.typing.MatLike":
    cap = cv2.VideoCapture(str(SAMPLE_VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    ok, frame = cap.read()
    cap.release()
    assert ok, "could not read frame 100 from the sample video"
    return frame


def test_class_names_read_from_weights_not_hardcoded() -> None:
    model = UltralyticsDetectionModel(str(DETECT_WEIGHTS))
    assert model.class_names == {0: "ball", 1: "cone", 2: "cone_1"}


def test_reproduces_measured_detections_on_frame_100(frame_100) -> None:
    """Frame 100 of ex22_Mayur.mov is the frame the plan's measurement was
    taken on: one ball, three cone-class boxes including the documented
    foot-false-positive at ~0.336 confidence, ~0.112 BL from an ankle."""
    model = UltralyticsDetectionModel(str(DETECT_WEIGHTS))
    detections = model.predict(frame_100, conf=0.25, iou=0.70)

    by_class: dict[str, list] = {}
    for d in detections:
        by_class.setdefault(d.class_name, []).append(d)

    assert "ball" in by_class and len(by_class["ball"]) == 1
    assert by_class["ball"][0].confidence == pytest.approx(0.806, abs=0.02)

    cones = by_class.get("cone", [])
    assert len(cones) == 3
    confidences = sorted(c.confidence for c in cones)
    assert confidences == pytest.approx([0.336, 0.659, 0.692], abs=0.02)

    # Coordinates must be normalized [0,1] of the ORIGINAL (not letterboxed) frame.
    for d in detections:
        assert 0.0 <= d.x1 < d.x2 <= 1.0
        assert 0.0 <= d.y1 < d.y2 <= 1.0
