"""PredictionQualityAnalyzer — runs every active rule against every
annotation on an image and returns triggered flags. Knows nothing about
cones or feet itself; see PLAN "The analyzer itself is generic... it runs
whatever rules are registered for a project."""
from __future__ import annotations

from dataclasses import dataclass

from app.models.project import Project
from app.services.quality.context import AnnotationLike, FrameContext
from app.services.quality.registry import get_active_rules


@dataclass(frozen=True)
class TriggeredFlag:
    annotation_id: str
    flag_type: str
    severity: float
    reason: str
    details: dict


class PredictionQualityAnalyzer:
    def analyze(
        self, annotations: list[AnnotationLike], context: FrameContext, project: Project
    ) -> list[TriggeredFlag]:
        triggered: list[TriggeredFlag] = []
        active_rules = get_active_rules(project)

        for det in annotations:
            for rule, params in active_rules:
                if not rule.applies_to(det.class_id):
                    continue
                result = rule.evaluate(det, context, params)
                if result is None:
                    continue
                if not result.details.get("applicable", True):
                    continue  # abstained — no person/track data available, not "passed"
                if result.flagged:
                    triggered.append(
                        TriggeredFlag(
                            annotation_id=det.id,
                            flag_type=rule.flag_type.value,
                            severity=result.severity,
                            reason=result.reason,
                            details=result.details,
                        )
                    )

        return triggered
