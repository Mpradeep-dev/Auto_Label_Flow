from __future__ import annotations

from pydantic import BaseModel


class PredictionOut(BaseModel):
    """One filtered detection, in the contract the spec's section 5 example
    describes — `source` is always "auto" here since this is model output
    that hasn't touched the annotation/review pipeline yet (Phase 3)."""

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    source: str = "auto"


class PredictResponse(BaseModel):
    model_id: str
    image_width: int
    image_height: int
    raw_count: int
    filtered_count: int
    predictions: list[PredictionOut]
