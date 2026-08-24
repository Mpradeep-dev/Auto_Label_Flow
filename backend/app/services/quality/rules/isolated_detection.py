"""ISOLATED_DETECTION — video only. Reuses SortTracker's own confirmation
threshold (a track needs `min_hits` matched detections before it's
"real") rather than inventing a second notion of "confirmed" — if the
tracker itself never trusted this detection enough to keep it alive, that
is the signal, not a separately-tuned constant."""
from __future__ import annotations

from app.models.quality import FlagType
from app.services.quality.context import AnnotationLike, FrameContext, RuleResult
from app.services.quality.registry import register_rule
from app.services.quality.rule_base import QualityRule

_MIN_HITS_DEFAULT = 5  # matches SortTracker's _STATIC_PREDICT_MIN_HITS for cone-like classes


class IsolatedDetectionRule(QualityRule):
    def evaluate(self, det: AnnotationLike, context: FrameContext, params: dict) -> RuleResult | None:
        if not context.is_video_frame:
            return None
        stats = context.track_stats_by_annotation_id.get(det.id)
        if stats is None:
            return RuleResult(flagged=False, severity=0.0, reason="No track data", details={"applicable": False})

        min_hits = params.get("min_hits", _MIN_HITS_DEFAULT)
        if stats.hits < min_hits:
            return RuleResult(
                flagged=True,
                severity=1 - stats.hits / min_hits,
                reason=f"Only matched {stats.hits} times across frames (needs {min_hits} to be a confirmed track)",
                details={"hits": stats.hits, "min_hits": min_hits},
            )
        return RuleResult(flagged=False, severity=0.0, reason="Track confirmed")


register_rule(
    IsolatedDetectionRule(
        flag_type=FlagType.ISOLATED_DETECTION,
        applies_to_classes=None,
        requires_temporal=True,
        default_params={"min_hits": _MIN_HITS_DEFAULT},
    )
)
