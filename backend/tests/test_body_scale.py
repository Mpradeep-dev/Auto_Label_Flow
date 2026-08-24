from __future__ import annotations

import pytest

from app.services.inference.detector import Keypoint
from app.services.quality.body_scale import compute_body_scale


def _kpts(overrides: dict[int, Keypoint]) -> list[Keypoint]:
    base = [Keypoint(x=0, y=0, confidence=0.0) for _ in range(17)]
    for idx, kp in overrides.items():
        base[idx] = kp
    return base


def test_prefers_leg_length_when_visible() -> None:
    kpts = _kpts(
        {
            11: Keypoint(x=0.1, y=0.4, confidence=0.9),
            15: Keypoint(x=0.1, y=0.7, confidence=0.9),
            12: Keypoint(x=0.2, y=0.4, confidence=0.9),
            16: Keypoint(x=0.2, y=0.7, confidence=0.9),
        }
    )
    scale, source = compute_body_scale(kpts)
    assert source == "leg"
    assert scale > 0


def test_falls_back_to_torso_when_legs_not_visible() -> None:
    kpts = _kpts(
        {
            5: Keypoint(x=0.1, y=0.2, confidence=0.9),
            6: Keypoint(x=0.2, y=0.2, confidence=0.9),
            11: Keypoint(x=0.1, y=0.4, confidence=0.9),
            12: Keypoint(x=0.2, y=0.4, confidence=0.9),
        }
    )
    scale, source = compute_body_scale(kpts)
    assert source == "torso"
    assert scale > 0


def test_falls_back_to_constant_when_nothing_visible() -> None:
    kpts = _kpts({})
    scale, source = compute_body_scale(kpts)
    assert source == "constant"
    assert scale == 0.4


def test_aspect_correction_changes_leg_length_for_non_square_frames() -> None:
    kpts = _kpts(
        {
            11: Keypoint(x=0.1, y=0.4, confidence=0.9),
            15: Keypoint(x=0.3, y=0.4, confidence=0.9),  # purely horizontal displacement
        }
    )
    scale_square, _ = compute_body_scale(kpts, aspect=1.0)
    scale_wide, _ = compute_body_scale(kpts, aspect=2.0)
    assert scale_wide == pytest.approx(scale_square * 2.0)


def test_low_visibility_keypoints_are_ignored() -> None:
    kpts = _kpts(
        {
            11: Keypoint(x=0.1, y=0.4, confidence=0.1),  # below default vis_threshold
            15: Keypoint(x=0.1, y=0.7, confidence=0.1),
        }
    )
    scale, source = compute_body_scale(kpts)
    assert source == "constant"
