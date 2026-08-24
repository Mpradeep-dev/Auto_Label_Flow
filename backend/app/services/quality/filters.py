"""Prediction filtering layer (PLAN spec section 5).

Deliberately NOT a confidence cutoff — low-confidence detections are review
candidates, not noise to discard (see PLAN "Measured numbers": the
documented foot-false-positive scored 0.336, below production's 0.55
stopgap; a platform meant to catch that failure mode must not filter it out
before a human ever sees it). What this layer DOES do:

  - drop detections below a floor so low the model itself is guessing
    (default 0.05 — well below any real signal, just numerical noise)
  - drop boxes outside sane size/aspect-ratio bounds (degenerate geometry,
    not a quality judgement)
  - suppress near-duplicate boxes of the SAME class, because detect_v1 is
    `end2end=True` (in-graph NMS) and duplicates survive it anyway —
    confirmed directly: ex22_Mayur.mov frame 140 has two `cone` boxes at
    the identical centre (conf 0.383 and 0.278) that the model's own NMS
    did not merge. Application-level dedup is required, not optional.

Confidence/suspicion-based triage (SUSPICIOUS_CONE, LOW_CONFIDENCE, etc.) is
a separate, richer layer — `services/quality/analyzer.py` (Phase 7) — that
flags rather than removes. This module only removes what is structurally
not a usable detection.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.services.inference.detector import Detection


@dataclass(frozen=True)
class FilterConfig:
    min_confidence: float = 0.05
    min_box_size: float = 0.003  # normalized width AND height floor — degenerate-box guard
    max_box_size: float = 0.95  # normalized width OR height ceiling — near-full-frame guard
    min_aspect_ratio: float = 0.05  # width/height
    max_aspect_ratio: float = 20.0
    duplicate_iou_threshold: float = 0.55
    duplicate_center_distance: float = 0.02  # normalized; catches identical-centre duplicates IoU might miss on tiny boxes


def _iou(a: Detection, b: Detection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = a.width * a.height
    area_b = b.width * b.height
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _center_distance(a: Detection, b: Detection) -> float:
    return ((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5


def _passes_geometry(det: Detection, config: FilterConfig) -> bool:
    if det.confidence < config.min_confidence:
        return False
    w, h = det.width, det.height
    if w <= 0 or h <= 0:
        return False
    if w < config.min_box_size and h < config.min_box_size:
        return False
    if w > config.max_box_size or h > config.max_box_size:
        return False
    aspect = w / h
    if not (config.min_aspect_ratio <= aspect <= config.max_aspect_ratio):
        return False
    return True


def _deduplicate(detections: list[Detection], config: FilterConfig) -> list[Detection]:
    """Greedy: sort by confidence descending, keep a box unless a
    higher-confidence box of the SAME class already kept is a near-duplicate
    (high IoU or near-identical centre). Cross-class overlap (e.g. a cone
    box overlapping a ball box) is left alone — that's a legitimate scene,
    not a duplicate."""
    by_class: dict[int, list[Detection]] = {}
    for det in detections:
        by_class.setdefault(det.class_id, []).append(det)

    kept: list[Detection] = []
    for class_id, dets in by_class.items():
        dets_sorted = sorted(dets, key=lambda d: d.confidence, reverse=True)
        class_kept: list[Detection] = []
        for det in dets_sorted:
            is_duplicate = any(
                _iou(det, k) >= config.duplicate_iou_threshold
                or _center_distance(det, k) <= config.duplicate_center_distance
                for k in class_kept
            )
            if not is_duplicate:
                class_kept.append(det)
        kept.extend(class_kept)
    return kept


def filter_predictions(detections: list[Detection], config: FilterConfig | None = None) -> list[Detection]:
    config = config or FilterConfig()
    geometry_ok = [d for d in detections if _passes_geometry(d, config)]
    return _deduplicate(geometry_ok, config)
