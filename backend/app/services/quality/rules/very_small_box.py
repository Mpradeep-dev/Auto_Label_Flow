"""VERY_SMALL_CONE — named for its origin (the spec's cone-degeneracy
case) but implemented generically: adaptive to the median box size of
SAME-CLASS boxes on the same image, same shape as `cone_validator.py`'s
adaptive-eps pattern in the sibling repo (median-size based, clipped) —
reused here rather than a fixed constant, since "small" only means
something relative to how big that class normally appears in this shot.
"""
from __future__ import annotations

import statistics

from app.models.quality import FlagType
from app.services.quality.context import AnnotationLike, FrameContext, RuleResult
from app.services.quality.registry import register_rule
from app.services.quality.rule_base import QualityRule

_RELATIVE_THRESHOLD = 0.4  # flag if area < 0.4x the same-class median on this image
_ABSOLUTE_MIN_WIDTH = 0.015  # normalized; fallback when there aren't enough same-class boxes to get a median


class VerySmallBoxRule(QualityRule):
    def evaluate(self, det: AnnotationLike, context: FrameContext, params: dict) -> RuleResult | None:
        relative_threshold = params.get("relative_threshold", _RELATIVE_THRESHOLD)
        absolute_min_width = params.get("absolute_min_width", _ABSOLUTE_MIN_WIDTH)

        same_class = [a for a in context.all_annotations if a.class_id == det.class_id]
        areas = [a.width * a.height for a in same_class]

        if len(same_class) >= 3:
            median_area = statistics.median(areas)
            area = det.width * det.height
            if median_area > 0 and area < relative_threshold * median_area:
                return RuleResult(
                    flagged=True,
                    severity=min(1.0, 1 - area / (relative_threshold * median_area)),
                    reason=f"Box area is {area / median_area:.0%} of the median {det.class_name} size on this image",
                    details={"area": area, "median_area": median_area},
                )
            return RuleResult(flagged=False, severity=0.0, reason="Size within normal range for this image")

        if det.width < absolute_min_width:
            return RuleResult(
                flagged=True,
                severity=min(1.0, 1 - det.width / absolute_min_width),
                reason=f"Box width {det.width:.4f} is below the absolute floor {absolute_min_width} (too few {det.class_name} boxes on this image for a median comparison)",
                details={"width": det.width, "absolute_min_width": absolute_min_width},
            )
        return RuleResult(flagged=False, severity=0.0, reason="Size above absolute floor")


register_rule(
    VerySmallBoxRule(
        flag_type=FlagType.VERY_SMALL_CONE,
        applies_to_classes=None,
        default_params={"relative_threshold": _RELATIVE_THRESHOLD, "absolute_min_width": _ABSOLUTE_MIN_WIDTH},
    )
)
