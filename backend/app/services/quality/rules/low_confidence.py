"""LOW_CONFIDENCE — class-agnostic, always active. Threshold is looked up
by class NAME substring, not a hardcoded class id, so the calibrated
"cone" band (0.65 — see PLAN "Measured numbers": real cones on detect_v1
sit at 0.67-0.79, so a generic 0.5 threshold would sit BELOW the entire
failure band and flag nothing) still applies if a different model's class
0 happens to also be named "cone", while any other class name falls back
to a conservative generic floor."""
from __future__ import annotations

from app.models.quality import FlagType
from app.services.quality.context import AnnotationLike, FrameContext, RuleResult
from app.services.quality.registry import register_rule
from app.services.quality.rule_base import QualityRule

_DEFAULT_THRESHOLD = 0.40
_NAME_THRESHOLDS = {"cone": 0.65, "cone_1": 0.65}  # calibrated for detect_v1 — see PLAN


class LowConfidenceRule(QualityRule):
    def evaluate(self, det: AnnotationLike, context: FrameContext, params: dict) -> RuleResult | None:
        if det.confidence is None:
            return None
        thresholds = params.get("name_thresholds", _NAME_THRESHOLDS)
        default = params.get("default_threshold", _DEFAULT_THRESHOLD)
        threshold = thresholds.get(det.class_name.lower(), default)
        if det.confidence < threshold:
            return RuleResult(
                flagged=True,
                severity=min(1.0, (threshold - det.confidence) / threshold),
                reason=f"Confidence {det.confidence:.2f} is below the {threshold:.2f} threshold for {det.class_name}",
                details={"confidence": det.confidence, "threshold": threshold},
            )
        return RuleResult(flagged=False, severity=0.0, reason="Confidence above threshold")


register_rule(
    LowConfidenceRule(
        flag_type=FlagType.LOW_CONFIDENCE,
        applies_to_classes=None,
        default_params={"default_threshold": _DEFAULT_THRESHOLD, "name_thresholds": _NAME_THRESHOLDS},
    )
)
