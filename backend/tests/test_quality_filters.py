from __future__ import annotations

from app.services.inference.detector import Detection
from app.services.quality.filters import FilterConfig, filter_predictions


def _det(class_id=1, conf=0.7, x1=0.1, y1=0.1, x2=0.2, y2=0.2) -> Detection:
    return Detection(class_id=class_id, class_name="cone", confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2)


def test_keeps_well_formed_detection() -> None:
    out = filter_predictions([_det()])
    assert len(out) == 1


def test_drops_degenerate_zero_area_box() -> None:
    out = filter_predictions([_det(x1=0.1, y1=0.1, x2=0.1, y2=0.2)])
    assert out == []


def test_drops_below_noise_floor_confidence() -> None:
    out = filter_predictions([_det(conf=0.01)])
    assert out == []


def test_keeps_low_confidence_above_noise_floor() -> None:
    """The whole point of this layer: low confidence is a review signal,
    not grounds for deletion. 0.336 is the measured foot-FP confidence on
    ex22_Mayur.mov frame 100 — it must survive filtering so a human sees it."""
    out = filter_predictions([_det(conf=0.336)])
    assert len(out) == 1
    assert out[0].confidence == 0.336


def test_drops_extreme_aspect_ratio() -> None:
    out = filter_predictions([_det(x1=0.0, y1=0.0, x2=0.9, y2=0.01)])  # 90:1 sliver
    assert out == []


def test_deduplicates_identical_centre_same_class() -> None:
    """Reproduces the measured case: ex22_Mayur.mov frame 140 has two `cone`
    boxes at the identical centre (conf 0.383, 0.278) that detect_v1's own
    in-graph NMS (end2end=True) did not merge."""
    a = _det(conf=0.383, x1=0.18, y1=0.60, x2=0.20, y2=0.63)
    b = _det(conf=0.278, x1=0.18, y1=0.60, x2=0.20, y2=0.63)
    out = filter_predictions([a, b])
    assert len(out) == 1
    assert out[0].confidence == 0.383  # keeps the higher-confidence box


def test_high_iou_duplicates_are_merged() -> None:
    a = _det(conf=0.7, x1=0.10, y1=0.10, x2=0.30, y2=0.30)
    b = _det(conf=0.6, x1=0.11, y1=0.11, x2=0.31, y2=0.31)  # heavy overlap
    out = filter_predictions([a, b])
    assert len(out) == 1


def test_does_not_deduplicate_across_classes() -> None:
    a = Detection(class_id=1, class_name="cone", confidence=0.7, x1=0.1, y1=0.1, x2=0.3, y2=0.3)
    b = Detection(class_id=0, class_name="ball", confidence=0.6, x1=0.1, y1=0.1, x2=0.3, y2=0.3)
    out = filter_predictions([a, b])
    assert len(out) == 2


def test_does_not_deduplicate_distinct_real_cones() -> None:
    """Two genuinely separate cones (measured 1.538 and 12.181 BL apart on
    ex22_Mayur.mov frame 100) must both survive."""
    a = _det(conf=0.692, x1=0.20, y1=0.58, x2=0.23, y2=0.64)
    b = _det(conf=0.659, x1=0.80, y1=0.62, x2=0.83, y2=0.68)
    out = filter_predictions([a, b])
    assert len(out) == 2


def test_custom_config_thresholds_apply() -> None:
    strict = FilterConfig(min_confidence=0.5)
    out = filter_predictions([_det(conf=0.336)], strict)
    assert out == []
