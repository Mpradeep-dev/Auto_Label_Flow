"""Runs SortTracker over one video's frames in order, using the already-
persisted annotations as the per-frame detections, and returns each
annotation's final track statistics — the input ISOLATED_DETECTION and
TEMPORAL_ANOMALY read. Runs on the ALREADY-PERSISTED annotation set
(not raw model output) so it reflects whatever a human has already
corrected, and is decoupled from inference timing (PLAN Phase 7 build
order treats quality analysis as a distinct step from auto-annotation).
"""
from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.image import Image
from app.services.annotation.service import list_annotations_for_image
from app.services.quality.context import TrackStats
from app.services.quality.sort_tracker import SortTracker


def run_temporal_pass_for_video(
    db: Session,
    *,
    video_id: uuid.UUID,
    static_class_ids: frozenset[int] = frozenset(),
    fast_class_ids: frozenset[int] = frozenset(),
) -> dict[uuid.UUID, TrackStats]:
    frames = list(
        db.scalars(
            select(Image).where(Image.video_id == video_id).order_by(Image.frame_index)
        )
    )
    if not frames:
        return {}

    tracker = SortTracker(static_class_ids=static_class_ids, fast_class_ids=fast_class_ids)
    track_id_by_annotation_id: dict[uuid.UUID, int] = {}

    for frame in frames:
        annotations = list_annotations_for_image(db, frame.id)
        detections = [
            {
                "class_id": a.class_id,
                "confidence": a.confidence or 0.0,
                "x1": a.x1,
                "y1": a.y1,
                "x2": a.x2,
                "y2": a.y2,
                "cx": (a.x1 + a.x2) / 2,
                "cy": (a.y1 + a.y2) / 2,
                "_annotation_id": a.id,
            }
            for a in annotations
        ]
        result = tracker.update(detections)
        for det in result:
            ann_id = det.get("_annotation_id")
            track_id = det.get("track_id")
            if ann_id is not None and track_id is not None:
                track_id_by_annotation_id[ann_id] = track_id

    stats_by_track_id = tracker.track_stats()
    return {
        ann_id: TrackStats(**{k: v for k, v in stats_by_track_id[track_id].items() if k != "class_id"})
        for ann_id, track_id in track_id_by_annotation_id.items()
        if track_id in stats_by_track_id
    }
