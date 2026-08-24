"""QualityRule — the interface every quality heuristic implements. The
analyzer (analyzer.py) knows nothing about cones or feet; it just runs
whatever rules are registered and applicable. See PLAN "PredictionQualityAnalyzer
— a rule registry, not a function"."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.models.quality import FlagType
from app.services.quality.context import AnnotationLike, FrameContext, RuleResult


@dataclass
class QualityRule(ABC):
    flag_type: FlagType
    applies_to_classes: tuple[int, ...] | None = None  # None = every class
    requires_pose: bool = False
    requires_temporal: bool = False
    # Non-None marks this rule as belonging to an optional, project-toggleable
    # pack (PLAN: "CONE_NEAR_PLAYER and SUSPICIOUS_CONE are shipped as a
    # concrete, optional, pluggable rule pack"). Class-agnostic rules leave
    # this None and are always active.
    pack_name: str | None = None
    default_params: dict = field(default_factory=dict)

    def applies_to(self, class_id: int) -> bool:
        return self.applies_to_classes is None or class_id in self.applies_to_classes

    @abstractmethod
    def evaluate(self, det: AnnotationLike, context: FrameContext, params: dict) -> RuleResult | None:
        """Return None if the rule doesn't apply to this detection at all
        (wrong class). Return a RuleResult with flagged=False for "checked,
        not suspicious". A pose-dependent rule that finds no person context
        available should return RuleResult(flagged=False, severity=0,
        reason="no person detected", details={"applicable": False}) — the
        analyzer treats `details.get("applicable", True) is False` as
        abstention, not clearance."""
