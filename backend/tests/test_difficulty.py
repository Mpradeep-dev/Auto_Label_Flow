from __future__ import annotations

from app.services.review.difficulty import DifficultyInputs, compute_difficulty


def test_clean_high_confidence_image_scores_low() -> None:
    score, _ = compute_difficulty(DifficultyInputs(confidences=[0.9, 0.85, 0.92]))
    assert score < 0.1


def test_suspicious_flag_dominates_the_score() -> None:
    score, components = compute_difficulty(DifficultyInputs(flag_severities=[0.9], confidences=[0.8]))
    assert score > 0.2
    assert components["flag_severity"] > components["low_confidence"]


def test_temporal_anomaly_adds_a_fixed_component() -> None:
    without, _ = compute_difficulty(DifficultyInputs(confidences=[0.8]))
    with_anomaly, _ = compute_difficulty(DifficultyInputs(confidences=[0.8], has_temporal_anomaly=True))
    assert with_anomaly > without


def test_pose_unavailable_adds_a_small_bump_not_a_dominant_signal() -> None:
    score, components = compute_difficulty(DifficultyInputs(pose_unavailable=True))
    assert 0 < score < 0.2
    assert components["pose_unavailable"] == 1.0


def test_score_is_always_clamped_to_one() -> None:
    score, _ = compute_difficulty(
        DifficultyInputs(
            flag_severities=[1.0, 1.0, 1.0],
            confidences=[0.0, 0.0],
            has_temporal_anomaly=True,
            pose_unavailable=True,
            prior_correction_count=50,
        )
    )
    assert score == 1.0


def test_empty_inputs_score_zero() -> None:
    score, components = compute_difficulty(DifficultyInputs())
    assert score == 0.0
    assert all(v == 0.0 for v in components.values())
