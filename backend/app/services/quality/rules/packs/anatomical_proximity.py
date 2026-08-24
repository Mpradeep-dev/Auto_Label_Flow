"""The anatomical_proximity pack — CONE_NEAR_PLAYER and SUSPICIOUS_CONE.
Optional, pluggable, project-scoped (PLAN Decision 1 / Core design
decision 1): activates only when a project has a pose auxiliary model
configured and targets whichever class ids its `quality_rule_config`
names (defaulting to any class whose name contains "cone" — see
registry.py). Proof the rule framework handles a real, hard case; not
evidence the framework assumes it.

Thresholds are calibrated against a live measurement on ex22_Mayur.mov
frame 100 (detect_v1 + pose_v1), reproduced directly while building this
platform — see PLAN "I reproduced this live before planning around it":

    foot false-positive cone: 0.112 BL from the nearest ankle
    lowest genuine cone observed in the same clip: 1.538 BL

CONE_NEAR_PLAYER's 0.75 BL threshold sits 6.7x above the measured false
positive and 2x below the lowest observed genuine cone — deliberately
conservative toward "flag more, let the human decide".
"""
from __future__ import annotations

from app.models.quality import FlagType
from app.services.quality.context import AnnotationLike, FrameContext, RuleResult
from app.services.quality.geometry import distance
from app.services.quality.registry import register_rule
from app.services.quality.rule_base import QualityRule

_CONE_NEAR_PLAYER_BL = 0.75
_LOWEST_OBSERVED_GENUINE_CONE_BL = 1.538  # denominator for the severity score, not a hard cutoff

_LEFT_ANKLE, _RIGHT_ANKLE = 15, 16
_ANKLE_VIS_THRESHOLD = 0.5


def _nearest_ankle_distance_bl(det: AnnotationLike, context: FrameContext) -> tuple[float, str] | None:
    """Returns (distance_in_body_lengths, body_scale_source) using the
    nearest visible ankle across every detected person, or None if no
    person/ankle is available at all (the analyzer must treat that as
    abstention, not clearance)."""
    best: tuple[float, str] | None = None
    for person in context.persons:
        for idx in (_LEFT_ANKLE, _RIGHT_ANKLE):
            if idx >= len(person.keypoints):
                continue
            ankle = person.keypoints[idx]
            if ankle.confidence < _ANKLE_VIS_THRESHOLD:
                continue
            raw_dist = distance((det.cx, det.cy), (ankle.x, ankle.y), context.aspect)
            dist_bl = raw_dist / max(person.body_scale, 1e-6)
            if best is None or dist_bl < best[0]:
                best = (dist_bl, person.body_scale_source)
    return best


class ConeNearPlayerRule(QualityRule):
    def evaluate(self, det: AnnotationLike, context: FrameContext, params: dict) -> RuleResult | None:
        if det.class_id not in params.get("target_class_ids", []):
            return None
        if not context.pose_available or not context.persons:
            return RuleResult(
                flagged=False, severity=0.0, reason="No person detected in this image", details={"applicable": False}
            )

        result = _nearest_ankle_distance_bl(det, context)
        if result is None:
            return RuleResult(
                flagged=False, severity=0.0, reason="No visible ankle keypoint", details={"applicable": False}
            )
        dist_bl, source = result
        threshold = params.get("threshold_bl", _CONE_NEAR_PLAYER_BL)
        if dist_bl <= threshold:
            return RuleResult(
                flagged=True,
                severity=max(0.0, 1 - dist_bl / _LOWEST_OBSERVED_GENUINE_CONE_BL),
                reason=f"{dist_bl:.2f} body-lengths from the nearest ankle (threshold {threshold})",
                details={"distance_bl": dist_bl, "threshold_bl": threshold, "body_scale_source": source},
            )
        return RuleResult(
            flagged=False,
            severity=0.0,
            reason=f"{dist_bl:.2f} body-lengths from the nearest ankle — beyond the threshold",
            details={"distance_bl": dist_bl, "body_scale_source": source},
        )


class SuspiciousConeRule(QualityRule):
    """Composite: CONE_NEAR_PLAYER's own predicate, restricted further to
    boxes whose centre falls in the lower third of a person's bounding
    box — narrowing "near the player" to "near the player's feet
    specifically", which is what the measured foot-false-positive case
    actually looks like."""

    def evaluate(self, det: AnnotationLike, context: FrameContext, params: dict) -> RuleResult | None:
        if det.class_id not in params.get("target_class_ids", []):
            return None
        if not context.pose_available or not context.persons:
            return RuleResult(
                flagged=False, severity=0.0, reason="No person detected in this image", details={"applicable": False}
            )

        result = _nearest_ankle_distance_bl(det, context)
        if result is None:
            return RuleResult(
                flagged=False, severity=0.0, reason="No visible ankle keypoint", details={"applicable": False}
            )
        dist_bl, source = result
        threshold = params.get("threshold_bl", _CONE_NEAR_PLAYER_BL)
        if dist_bl > threshold:
            return RuleResult(flagged=False, severity=0.0, reason="Not near a player", details={"distance_bl": dist_bl})

        in_lower_third = any(
            person.y1 + (person.y2 - person.y1) * (2 / 3) <= det.cy <= person.y2
            and person.x1 <= det.cx <= person.x2
            for person in context.persons
        )
        if not in_lower_third:
            return RuleResult(
                flagged=False,
                severity=0.0,
                reason="Near a player but not in the lower-body region",
                details={"distance_bl": dist_bl},
            )

        return RuleResult(
            flagged=True,
            severity=max(0.0, 1 - dist_bl / _LOWEST_OBSERVED_GENUINE_CONE_BL),
            reason=f"{dist_bl:.2f} body-lengths from the nearest ankle, inside the player's lower-body region — likely a foot, not a {det.class_name}",
            details={"distance_bl": dist_bl, "threshold_bl": threshold, "body_scale_source": source},
        )


register_rule(
    ConeNearPlayerRule(
        flag_type=FlagType.CONE_NEAR_PLAYER,
        applies_to_classes=None,
        requires_pose=True,
        pack_name="anatomical_proximity",
        default_params={"threshold_bl": _CONE_NEAR_PLAYER_BL},
    )
)
register_rule(
    SuspiciousConeRule(
        flag_type=FlagType.SUSPICIOUS_CONE,
        applies_to_classes=None,
        requires_pose=True,
        pack_name="anatomical_proximity",
        default_params={"threshold_bl": _CONE_NEAR_PLAYER_BL},
    )
)
