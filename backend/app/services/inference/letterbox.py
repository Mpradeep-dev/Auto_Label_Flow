"""Ported verbatim from gsp-video-ai-processing-service
`fcg-ai-video-processing/pipelines/letterbox.py`. YOLO-style letterbox
resizing: scale a frame to fit the model's square input size without
distorting its aspect ratio, padding the remainder. `LetterboxMeta` records
the scale/padding applied so detection boxes produced on the resized frame
can be mapped back to original-frame pixel coordinates.

Kept byte-for-byte identical to the source so this platform's coordinate
frame matches the production pipeline's exactly — any divergence here would
make measured thresholds (see PLAN "Measured numbers") silently wrong.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxMeta:
    """Records how a frame was letterboxed, so coordinates can be mapped back to original pixels."""
    orig_width: int
    orig_height: int
    input_width: int
    input_height: int
    scale: float
    pad_x: float
    pad_y: float


def letterbox(
    frame: np.ndarray,
    size: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, LetterboxMeta]:
    """YOLO-style letterbox resize without aspect-ratio distortion."""
    input_w, input_h = size
    orig_h, orig_w = frame.shape[:2]
    scale = min(input_w / orig_w, input_h / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_h, input_w, 3), color, dtype=frame.dtype)
    pad_x = (input_w - new_w) / 2
    pad_y = (input_h - new_h) / 2
    x0 = int(pad_x)
    y0 = int(pad_y)
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

    return canvas, LetterboxMeta(
        orig_width=orig_w,
        orig_height=orig_h,
        input_width=input_w,
        input_height=input_h,
        scale=scale,
        pad_x=float(x0),
        pad_y=float(y0),
    )


def keypoints_to_original(xy: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """Map keypoints from letterboxed inference pixels back to original pixels."""
    out = xy.astype(np.float32, copy=True)
    out[:, 0] = (out[:, 0] - meta.pad_x) / meta.scale
    out[:, 1] = (out[:, 1] - meta.pad_y) / meta.scale
    out[:, 0] = np.clip(out[:, 0], 0, meta.orig_width - 1)
    out[:, 1] = np.clip(out[:, 1], 0, meta.orig_height - 1)
    return out


def boxes_to_original(xyxy: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """Map xyxy boxes from letterboxed inference pixels back to original pixels."""
    out = xyxy.astype(np.float32, copy=True)
    out[:, [0, 2]] = (out[:, [0, 2]] - meta.pad_x) / meta.scale
    out[:, [1, 3]] = (out[:, [1, 3]] - meta.pad_y) / meta.scale
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, meta.orig_width - 1)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, meta.orig_height - 1)
    return out
