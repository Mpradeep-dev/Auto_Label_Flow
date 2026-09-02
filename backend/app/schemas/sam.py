"""SAM-assisted segmentation: the interactive-prompt request/response, and
the optional-checkpoint status the Settings page polls (mirrors
`schemas/integration.py`'s job-status shapes for the add-on packs)."""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class SamModelStatusRead(BaseModel):
    name: str
    label: str
    blurb: str
    installed: bool
    size_bytes: int | None = None


class SegmentRequest(BaseModel):
    """One or more prompt points for SAM, normalized [0,1] image
    coordinates. `labels[i]` is 1 (foreground) or 0 (background) for
    `points[i]` — defaults to all-foreground, since the canvas's v1
    interaction is a single foreground click per segment."""

    variant: str = Field(min_length=1)
    points: list[tuple[float, float]] = Field(min_length=1)
    labels: list[int] | None = None

    @model_validator(mode="after")
    def _labels_match_points(self) -> "SegmentRequest":
        if self.labels is not None and len(self.labels) != len(self.points):
            raise ValueError("labels must be the same length as points when given")
        return self

    @property
    def resolved_labels(self) -> list[int]:
        return self.labels if self.labels is not None else [1] * len(self.points)


class SegmentResponse(BaseModel):
    """`points` is `None` when SAM produced no usable mask for this prompt
    — the caller shows "try clicking elsewhere", not an error."""

    points: list[list[float]] | None
