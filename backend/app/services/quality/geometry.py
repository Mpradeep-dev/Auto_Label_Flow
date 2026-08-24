"""Ported verbatim from gsp-video-ai-processing-service
`fcg-ai-video-processing/core/geometry.py`. Aspect-ratio-aware geometry
primitives for distance math over normalized [0,1] image coordinates.

Why this exists: every coordinate in this codebase is normalized by
dividing raw pixel x by frame WIDTH and pixel y by frame HEIGHT
independently. For a non-square frame (16:9 landscape, 9:16 portrait phone
footage — the overwhelming majority of real recordings) a plain Euclidean
`hypot` on raw (x, y) pairs is geometrically distorted. The fix: multiply
the x-component by the frame's aspect ratio (width / height) before doing
any Euclidean geometry. This is the exact correction the measured 0.112 BL
/ 1.538 BL numbers in the plan were computed with — using uncorrected
distance here would silently produce different numbers than the ones the
anatomical_proximity thresholds were calibrated against.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def _xy(point) -> Tuple[float, float]:
    """Extract (x, y) from a point-like object (has .x/.y) or an (x, y) pair."""
    if hasattr(point, "x") and hasattr(point, "y"):
        return float(point.x), float(point.y)
    return float(point[0]), float(point[1])


def corrected_delta(dx: float, dy: float, aspect: float = 1.0) -> Tuple[float, float]:
    """Rescale a normalised-coordinate delta so both axes are in the same
    physical (y-normalised) units. `aspect` = frame width / frame height."""
    return dx * aspect, dy


def distance(a, b, aspect: float = 1.0) -> float:
    """Aspect-corrected Euclidean distance between two normalised-coordinate points."""
    ax, ay = _xy(a)
    bx, by = _xy(b)
    dx, dy = corrected_delta(ax - bx, ay - by, aspect)
    return float(np.hypot(dx, dy))
