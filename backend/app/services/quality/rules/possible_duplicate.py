"""POSSIBLE_DUPLICATE — class-agnostic, always active. The filtering layer
(services/quality/filters.py) already removes near-duplicates produced by
a SINGLE inference call, but persisted annotations can still collide from
other sources: a second auto-annotation run without replace, a human
accidentally drawing over an existing box, or (the case this was measured
against) `end2end=True` in-graph NMS not catching everything — ex22_Mayur.mov
frame 140 had two identical-centre `cone` boxes at 0.383/0.278 confidence
that detect_v1's own NMS left untouched. This rule flags for review rather
than silently removing, since a flagged pair might legitimately be two
separate close objects."""
from __future__ import annotations

from app.models.quality import FlagType
from app.services.quality.context import AnnotationLike, FrameContext, RuleResult
from app.services.quality.registry import register_rule
from app.services.quality.rule_base import QualityRule

_IOU_THRESHOLD = 0.55
_CENTER_DISTANCE_BL = 1.0  # falls back to normalized units when body_scale isn't available


def _iou(a: AnnotationLike, b: AnnotationLike) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union > 0 else 0.0


class PossibleDuplicateRule(QualityRule):
    def evaluate(self, det: AnnotationLike, context: FrameContext, params: dict) -> RuleResult | None:
        iou_threshold = params.get("iou_threshold", _IOU_THRESHOLD)
        center_distance = params.get("center_distance_bl", _CENTER_DISTANCE_BL)
        body_scale = context.persons[0].body_scale if context.persons else 1.0

        best_iou = 0.0
        best_dist_bl = float("inf")
        for other in context.all_annotations:
            if other.id == det.id or other.class_id != det.class_id:
                continue
            iou = _iou(det, other)
            dist_bl = ((det.cx - other.cx) ** 2 + (det.cy - other.cy) ** 2) ** 0.5 / max(body_scale, 1e-6)
            best_iou = max(best_iou, iou)
            best_dist_bl = min(best_dist_bl, dist_bl)

        if best_iou >= iou_threshold or best_dist_bl <= center_distance:
            return RuleResult(
                flagged=True,
                severity=max(best_iou, 1 - min(1.0, best_dist_bl / max(center_distance, 1e-6))),
                reason=f"Overlaps another {det.class_name} box (IoU {best_iou:.2f})",
                details={"best_iou": best_iou, "best_center_distance_bl": best_dist_bl},
            )
        return RuleResult(flagged=False, severity=0.0, reason="No overlapping same-class box found")


register_rule(
    PossibleDuplicateRule(
        flag_type=FlagType.POSSIBLE_DUPLICATE,
        applies_to_classes=None,
        default_params={"iou_threshold": _IOU_THRESHOLD, "center_distance_bl": _CENTER_DISTANCE_BL},
    )
)
