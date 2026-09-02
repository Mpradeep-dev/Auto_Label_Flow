"""SAM-assisted interactive segmentation. Alongside `ultralytics_adapter.py`,
this is the only module that imports Ultralytics directly — nothing outside
`services/inference/` may (see `detector.py`'s docstring).

A SAM prompt (one or more foreground/background points) produces a mask;
this module traces that mask into the same single-outer-ring polygon
convention every other polygon path in the app already uses
(`services/dataset/coco_common.polygon_points_from_segmentation`) — so a
SAM-assisted shape is indistinguishable in storage from a hand-drawn one.
No new `ShapeType`, no raster storage.
"""
from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

from app.services.inference.detector import ModelLoadError, require_weights_file


class SamSegmentationError(RuntimeError):
    """SAM loaded fine but returned nothing usable for this prompt — a
    caller-facing 4xx, not a crash."""


def _resolve_device() -> str:
    """Mirrors `ultralytics_adapter._resolve_device` — duplicated rather
    than imported, since that helper is private to a sibling module and this
    is a two-line check, not worth a shared-utility indirection for."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:
        pass
    return "cpu"


def _mask_to_polygon(mask: np.ndarray, width: int, height: int) -> list[list[float]] | None:
    """A boolean/uint8 mask (at the original image's resolution) -> a
    normalized `[[x, y], ...]` outer-ring polygon, largest contour only —
    same "single outer ring, no holes" convention as every COCO/CVAT-derived
    polygon in this app. `None` if the mask has no usable contour."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 3:
        return None
    points = [
        [min(1.0, max(0.0, float(x) / width)), min(1.0, max(0.0, float(y) / height))]
        for [[x, y]] in largest
    ]
    # De-duplicate consecutive identical points (cv2 can emit them at corners)
    # without collapsing the ring below the 3-point minimum every polygon
    # consumer (create_annotation's _bbox_from_points) requires.
    deduped = [points[0]] + [p for prev, p in zip(points, points[1:]) if p != prev]
    return deduped if len(deduped) >= 3 else None


class SamSegmenter:
    """Wraps one loaded `ultralytics.SAM` checkpoint. Construction is the
    expensive part (weights load onto CPU/GPU) — see `get_segmenter` below
    for the process-level cache that avoids paying it per request."""

    def __init__(self, weights_path: str) -> None:
        require_weights_file(Path(weights_path))
        from ultralytics import SAM

        try:
            self._model = SAM(weights_path)
        except Exception as exc:  # pragma: no cover - depends on corrupt weights, hard to fixture
            raise ModelLoadError(f"Failed to load SAM weights at {weights_path}: {exc}") from exc
        self._device = _resolve_device()

    def segment(
        self, image: np.ndarray, points: list[tuple[float, float]], labels: list[int]
    ) -> list[list[float]] | None:
        """`points` are normalized [0,1] image coordinates (foreground click
        points, typically); `labels` is 1 (foreground) or 0 (background) per
        point, same length as `points`. Returns a normalized polygon, or
        `None` if SAM produced no usable mask for this prompt."""
        if not points or len(points) != len(labels):
            raise SamSegmentationError("points and labels must be the same non-empty length")

        height, width = image.shape[:2]
        px_points = [[int(round(x * width)), int(round(y * height))] for x, y in points]

        results = self._model.predict(
            source=image, points=[px_points], labels=[labels], device=self._device, verbose=False
        )
        if not results or results[0].masks is None or len(results[0].masks.data) == 0:
            return None

        result = results[0]
        # Ultralytics already traces mask contours for its own seg-label
        # export (`Results.masks.xy`, pixel coordinates at the ORIGINAL
        # image's resolution) — prefer that over re-deriving it, and only
        # fall back to findContours on the raw mask array if that attribute
        # is absent (older/atypical result shapes).
        xy = getattr(result.masks, "xy", None)
        if xy is not None and len(xy) > 0 and len(xy[0]) >= 3:
            points_px = xy[0]
            return _mask_to_polygon_from_points(points_px, width, height)

        mask = result.masks.data[0].cpu().numpy()
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        return _mask_to_polygon(mask, width, height)


def _mask_to_polygon_from_points(points_px, width: int, height: int) -> list[list[float]] | None:
    if len(points_px) < 3:
        return None
    normalized = [
        [min(1.0, max(0.0, float(x) / width)), min(1.0, max(0.0, float(y) / height))] for x, y in points_px
    ]
    deduped = [normalized[0]] + [p for prev, p in zip(normalized, normalized[1:]) if p != prev]
    return deduped if len(deduped) >= 3 else None


_segmenter_cache: dict[str, SamSegmenter] = {}
_cache_lock = threading.Lock()


def get_segmenter(weights_path: str) -> SamSegmenter:
    """Process-level load-once cache, same rationale as
    `registry.py`'s detection/pose caches — a SAM checkpoint is expensive to
    load and this app's `gpu` queue runs at concurrency=1 regardless."""
    cached = _segmenter_cache.get(weights_path)
    if cached is not None:
        return cached
    with _cache_lock:
        cached = _segmenter_cache.get(weights_path)
        if cached is None:
            cached = SamSegmenter(weights_path)
            _segmenter_cache[weights_path] = cached
        return cached


def invalidate_segmenter(weights_path: str) -> None:
    """Drop a cached segmenter — called when its weights file is removed, so
    a later re-download at the same path doesn't silently keep serving the
    (now-deleted) old in-memory model for the rest of this process's life."""
    with _cache_lock:
        _segmenter_cache.pop(weights_path, None)
