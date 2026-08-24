from __future__ import annotations

from pydantic import BaseModel


class AutoLabelAcceptance(BaseModel):
    total_auto_predictions: int
    accepted: int
    corrected: int
    rejected: int
    acceptance_rate: float | None


class DatasetStatistics(BaseModel):
    total_images: int
    reviewed_images: int
    pending_images: int
    completion_pct: float
    total_annotations: int
    annotations_by_class: dict[str, int]
    annotations_by_source: dict[str, int]
    average_confidence: float | None
    low_confidence_predictions: int
    suspicious_cones: int
    auto_label_acceptance: AutoLabelAcceptance


class ErrorAnalysis(BaseModel):
    total_categorized_deletions: int
    by_category: dict[str, int]
    by_reason: dict[str, int]


class ModelMetricsUpdate(BaseModel):
    metrics: dict
