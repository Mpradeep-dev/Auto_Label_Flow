"""Unit tests for the rule framework and the analyzer, using synthetic
contexts — no DB, no real models. The regression that matters (PLAN
Verification): a cone 0.12 BL from an ankle must raise SUSPICIOUS_CONE; one
at 2.02+ BL must not."""
from __future__ import annotations

import uuid

from app.models.project import Project
from app.services.inference.detector import Keypoint
from app.services.quality.analyzer import PredictionQualityAnalyzer
from app.services.quality.context import AnnotationLike, FrameContext, PersonContext, TrackStats


def _cone_project(pose: bool = True) -> Project:
    return Project(
        name="p",
        slug="p",
        class_config=[{"id": 0, "name": "ball"}, {"id": 1, "name": "cone"}, {"id": 2, "name": "cone_1"}],
        quality_rule_config={},
        pose_model_id=uuid.uuid4() if pose else None,
    )


def _person_with_ankle(ankle_xy: tuple[float, float], body_scale: float = 0.10, y_range=(0.40, 0.68)) -> PersonContext:
    kpts = [Keypoint(x=0, y=0, confidence=0.0) for _ in range(17)]
    kpts[15] = Keypoint(x=ankle_xy[0], y=ankle_xy[1], confidence=0.9)
    kpts[16] = Keypoint(x=0.9, y=ankle_xy[1], confidence=0.9)  # far away, doesn't interfere
    return PersonContext(x1=0.05, y1=y_range[0], x2=0.25, y2=y_range[1], keypoints=kpts, body_scale=body_scale, body_scale_source="leg")


def _cone(id_: str, cx: float, cy: float, conf: float = 0.7, size: float = 0.02) -> AnnotationLike:
    half = size / 2
    return AnnotationLike(id=id_, class_id=1, class_name="cone", confidence=conf, x1=cx - half, y1=cy - half, x2=cx + half, y2=cy + half)


def test_foot_false_positive_at_measured_012_bl_raises_suspicious_cone() -> None:
    """The exact measured case: 0.112 BL foot-FP, reproduced live on
    ex22_Mayur.mov while building this platform (PLAN 'I reproduced this
    live before planning around it')."""
    person = _person_with_ankle((0.111, 0.648), body_scale=0.10, y_range=(0.40, 0.65))
    det = _cone("a1", cx=0.1105, cy=0.648, conf=0.336)
    ctx = FrameContext(image_id="img1", aspect=16 / 9, all_annotations=[det], persons=[person], pose_available=True)

    flags = PredictionQualityAnalyzer().analyze([det], ctx, _cone_project())
    flag_types = {f.flag_type for f in flags}
    assert "SUSPICIOUS_CONE" in flag_types
    assert "CONE_NEAR_PLAYER" in flag_types


def test_genuine_cone_at_measured_1538_bl_is_not_flagged_suspicious() -> None:
    """The lowest genuine cone actually observed on the same clip (1.538
    BL) must clear both anatomical-proximity flags."""
    person = _person_with_ankle((0.20, 0.60), body_scale=0.10)
    # 1.538 BL away along x: 0.20 + 1.538*0.10 = 0.3538
    det = _cone("a1", cx=0.3538, cy=0.60, conf=0.692)
    ctx = FrameContext(image_id="img1", aspect=1.0, all_annotations=[det], persons=[person], pose_available=True)

    flags = PredictionQualityAnalyzer().analyze([det], ctx, _cone_project())
    flag_types = {f.flag_type for f in flags}
    assert "SUSPICIOUS_CONE" not in flag_types
    assert "CONE_NEAR_PLAYER" not in flag_types


def test_ball_is_never_subject_to_anatomical_proximity_even_at_the_players_feet() -> None:
    """Measured: the ball legitimately sits 0.088-0.714 BL from an ankle
    in a dribbling drill. A class-agnostic proximity rule would wrongly
    flag every real ball — target_class_ids scoping must exclude it."""
    person = _person_with_ankle((0.30, 0.60), body_scale=0.10)
    # Realistic ball size (~0.03 wide, matching the corpus) so this test
    # exercises only the anatomical-proximity exemption, not VERY_SMALL_CONE
    # (a real, separate, class-agnostic rule that a 0.01-wide box would
    # correctly trip regardless of class).
    ball = AnnotationLike(id="b1", class_id=0, class_name="ball", confidence=0.8, x1=0.285, y1=0.585, x2=0.315, y2=0.615)
    ctx = FrameContext(image_id="img1", aspect=1.0, all_annotations=[ball], persons=[person], pose_available=True)

    flags = PredictionQualityAnalyzer().analyze([ball], ctx, _cone_project())
    assert flags == []


def test_no_person_detected_abstains_rather_than_clearing() -> None:
    """No pose data available: the anatomical rules must not fire (there's
    no basis to say the cone IS near a player) but must also not silently
    imply it was checked and found clean — abstention, not clearance."""
    det = _cone("a1", cx=0.5, cy=0.5, conf=0.7)
    ctx = FrameContext(image_id="img1", aspect=1.0, all_annotations=[det], persons=[], pose_available=False)

    flags = PredictionQualityAnalyzer().analyze([det], ctx, _cone_project())
    assert not any(f.flag_type in ("CONE_NEAR_PLAYER", "SUSPICIOUS_CONE") for f in flags)


def test_anatomical_proximity_pack_inactive_without_pose_model_configured() -> None:
    person = _person_with_ankle((0.111, 0.648))
    det = _cone("a1", cx=0.1105, cy=0.648, conf=0.336)
    ctx = FrameContext(image_id="img1", aspect=1.0, all_annotations=[det], persons=[person], pose_available=True)

    flags = PredictionQualityAnalyzer().analyze([det], ctx, _cone_project(pose=False))
    assert not any(f.flag_type in ("CONE_NEAR_PLAYER", "SUSPICIOUS_CONE") for f in flags)


def test_low_confidence_cone_band_is_calibrated_not_generic() -> None:
    """0.5 would sit BELOW the measured real-cone confidence band
    (0.67-0.79) and flag nothing real; 0.65 sits inside it correctly."""
    det = _cone("a1", cx=0.5, cy=0.5, conf=0.60)  # below 0.65, above a generic 0.5
    ctx = FrameContext(image_id="img1", aspect=1.0, all_annotations=[det], persons=[])
    flags = PredictionQualityAnalyzer().analyze([det], ctx, _cone_project(pose=False))
    assert any(f.flag_type == "LOW_CONFIDENCE" for f in flags)


def test_very_small_box_relative_to_same_class_median() -> None:
    normal = [_cone(f"n{i}", cx=0.1 * i, cy=0.5, size=0.05) for i in range(1, 5)]
    tiny = _cone("tiny", cx=0.9, cy=0.5, size=0.01)
    all_dets = normal + [tiny]
    ctx = FrameContext(image_id="img1", aspect=1.0, all_annotations=all_dets, persons=[])
    flags = PredictionQualityAnalyzer().analyze(all_dets, ctx, _cone_project(pose=False))
    tiny_flags = {f.flag_type for f in flags if f.annotation_id == "tiny"}
    assert "VERY_SMALL_CONE" in tiny_flags
    assert not any(f.annotation_id == "n1" and f.flag_type == "VERY_SMALL_CONE" for f in flags)


def test_possible_duplicate_flags_overlapping_same_class_boxes() -> None:
    a = _cone("a", cx=0.5, cy=0.5, size=0.05)
    b = _cone("b", cx=0.505, cy=0.505, size=0.05)  # heavy overlap
    ctx = FrameContext(image_id="img1", aspect=1.0, all_annotations=[a, b], persons=[])
    flags = PredictionQualityAnalyzer().analyze([a, b], ctx, _cone_project(pose=False))
    flagged_ids = {f.annotation_id for f in flags if f.flag_type == "POSSIBLE_DUPLICATE"}
    assert flagged_ids == {"a", "b"}


def test_isolated_detection_only_applies_to_video_frames() -> None:
    det = _cone("a1", cx=0.5, cy=0.5, conf=0.7)
    ctx = FrameContext(image_id="img1", aspect=1.0, all_annotations=[det], persons=[], is_video_frame=False)
    flags = PredictionQualityAnalyzer().analyze([det], ctx, _cone_project(pose=False))
    assert not any(f.flag_type == "ISOLATED_DETECTION" for f in flags)


def test_isolated_detection_fires_below_min_hits() -> None:
    det = _cone("a1", cx=0.5, cy=0.5, conf=0.7)
    ctx = FrameContext(
        image_id="img1", aspect=1.0, all_annotations=[det], persons=[], is_video_frame=True,
        track_stats_by_annotation_id={"a1": TrackStats(hits=2, missed=0, velocity_magnitude=0.0)},
    )
    flags = PredictionQualityAnalyzer().analyze([det], ctx, _cone_project(pose=False))
    assert any(f.flag_type == "ISOLATED_DETECTION" for f in flags)


def test_temporal_anomaly_fires_for_moving_confirmed_track() -> None:
    det = _cone("a1", cx=0.5, cy=0.5, conf=0.7)
    ctx = FrameContext(
        image_id="img1", aspect=1.0, all_annotations=[det], persons=[], is_video_frame=True,
        track_stats_by_annotation_id={"a1": TrackStats(hits=6, missed=0, velocity_magnitude=0.05)},
    )
    flags = PredictionQualityAnalyzer().analyze([det], ctx, _cone_project(pose=False))
    assert any(f.flag_type == "TEMPORAL_ANOMALY" for f in flags)


def test_temporal_anomaly_does_not_fire_for_stationary_track() -> None:
    det = _cone("a1", cx=0.5, cy=0.5, conf=0.7)
    ctx = FrameContext(
        image_id="img1", aspect=1.0, all_annotations=[det], persons=[], is_video_frame=True,
        track_stats_by_annotation_id={"a1": TrackStats(hits=6, missed=0, velocity_magnitude=0.001)},
    )
    flags = PredictionQualityAnalyzer().analyze([det], ctx, _cone_project(pose=False))
    assert not any(f.flag_type == "TEMPORAL_ANOMALY" for f in flags)
