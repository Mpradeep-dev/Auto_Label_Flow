"""TEMPORAL_ANOMALY — video only. For classes the tracker treats as
"static" (project-configured — see services/quality/analyzer.py), a
track's Kalman velocity should be ~zero after confirmation; sustained
motion means whatever is being tracked isn't the static object it was
classified as (the concrete case: a "cone" track that's actually
attached to a moving foot). Cones are static by definition — a moving
"cone" is the anomaly."""
from __future__ import annotations

from app.models.quality import FlagType
from app.services.quality.context import AnnotationLike, FrameContext, RuleResult
from app.services.quality.registry import register_rule
from app.services.quality.rule_base import QualityRule

_VELOCITY_THRESHOLD = 0.02  # normalized units/frame


class TemporalAnomalyRule(QualityRule):
    def evaluate(self, det: AnnotationLike, context: FrameContext, params: dict) -> RuleResult | None:
        if not context.is_video_frame:
            return None
        stats = context.track_stats_by_annotation_id.get(det.id)
        if stats is None:
            return RuleResult(flagged=False, severity=0.0, reason="No track data", details={"applicable": False})

        threshold = params.get("velocity_threshold", _VELOCITY_THRESHOLD)
        if stats.hits >= 3 and stats.velocity_magnitude > threshold:
            return RuleResult(
                flagged=True,
                severity=min(1.0, stats.velocity_magnitude / (threshold * 5)),
                reason=f"Track is moving at {stats.velocity_magnitude:.3f} units/frame after confirmation — {det.class_name} is expected to be static",
                details={"velocity_magnitude": stats.velocity_magnitude, "threshold": threshold},
            )
        return RuleResult(flagged=False, severity=0.0, reason="Track stationary")


register_rule(
    TemporalAnomalyRule(
        flag_type=FlagType.TEMPORAL_ANOMALY,
        applies_to_classes=None,
        requires_temporal=True,
        default_params={"velocity_threshold": _VELOCITY_THRESHOLD},
    )
)
