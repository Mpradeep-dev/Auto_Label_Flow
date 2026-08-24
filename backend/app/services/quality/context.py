"""Shared context objects passed to every quality rule. Kept separate from
the rules themselves so a rule's signature stays `evaluate(det, context) ->
RuleResult | None` regardless of how much context-building machinery sits
behind `context` — see PLAN "FrameContext carries the pose result... This
is why pose_v1 runs alongside detect_v1 even though person is never itself
an annotation class."
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AnnotationLike:
    """Duck-typed view of one box under evaluation — works for both a live
    `Annotation` ORM row and a plain dict, so rules don't import the model
    layer directly."""

    id: str
    class_id: int
    class_name: str
    confidence: float | None
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2


@dataclass(frozen=True)
class PersonContext:
    x1: float
    y1: float
    x2: float
    y2: float
    keypoints: list  # list[Keypoint], COCO-ordered, index 0-16
    body_scale: float
    body_scale_source: str  # "leg" | "torso" | "constant"


@dataclass
class TrackStats:
    hits: int
    missed: int
    velocity_magnitude: float


@dataclass
class FrameContext:
    """Everything a rule might need beyond the single detection it's
    evaluating. `all_annotations` is every OTHER box on the same image
    (for duplicate/isolation checks); `persons` is empty when no pose
    model is configured for the project — rules must treat that as
    "inapplicable", not "zero suspicion" (PLAN: abstention is a distinct
    state from "passed")."""

    image_id: str
    aspect: float
    all_annotations: list[AnnotationLike] = field(default_factory=list)
    persons: list[PersonContext] = field(default_factory=list)
    pose_available: bool = False
    track_stats_by_annotation_id: dict[str, TrackStats] = field(default_factory=dict)
    is_video_frame: bool = False


@dataclass(frozen=True)
class RuleResult:
    flagged: bool
    severity: float  # 0-1
    reason: str
    details: dict = field(default_factory=dict)
