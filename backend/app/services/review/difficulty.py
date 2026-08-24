"""Active-learning difficulty score (PLAN "Difficulty score: stored
column, recomputed on write"). Weighted combination of signals, kept in
one config object so the formula is tunable without touching the
recompute call sites — see PLAN "the weights live in one config object so
they are tunable without touching query code."

Deliberately per-image, not per-dataset: computing "detection count vs
clip median" would need a whole-dataset aggregate query on every single
recompute (which happens after every inference write and every human
correction — see call sites), which doesn't scale. The flag-severity and
confidence components already capture "this image looks unusual" without
needing that cross-image comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DifficultyWeights:
    flag_severity: float = 0.5
    low_confidence: float = 0.2
    temporal_anomaly: float = 0.15
    pose_unavailable: float = 0.05
    prior_corrections: float = 0.10


DEFAULT_WEIGHTS = DifficultyWeights()


@dataclass
class DifficultyInputs:
    flag_severities: list[float] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    has_temporal_anomaly: bool = False
    pose_unavailable: bool = False
    prior_correction_count: int = 0  # corrections on OTHER images from the same video, recent window


def compute_difficulty(inputs: DifficultyInputs, weights: DifficultyWeights = DEFAULT_WEIGHTS) -> tuple[float, dict]:
    """Returns (score in [0,1], component breakdown) — the breakdown is
    stored separately from the combined score (PLAN: "so reweighting the
    formula later is a cheap recombination task, not a full geometry
    re-run")."""
    flag_component = min(1.0, sum(inputs.flag_severities) / 2) if inputs.flag_severities else 0.0
    confidence_component = (
        sum(1 - c for c in inputs.confidences) / len(inputs.confidences) if inputs.confidences else 0.0
    )
    temporal_component = 1.0 if inputs.has_temporal_anomaly else 0.0
    pose_component = 1.0 if inputs.pose_unavailable else 0.0
    correction_component = min(1.0, inputs.prior_correction_count / 5)

    components = {
        "flag_severity": flag_component,
        "low_confidence": confidence_component,
        "temporal_anomaly": temporal_component,
        "pose_unavailable": pose_component,
        "prior_corrections": correction_component,
    }
    score = (
        weights.flag_severity * flag_component
        + weights.low_confidence * confidence_component
        + weights.temporal_anomaly * temporal_component
        + weights.pose_unavailable * pose_component
        + weights.prior_corrections * correction_component
    )
    return min(1.0, score), components
