"""DetectionModel — the interface every detector implementation sits behind.
Nothing outside `services/inference/` may import Ultralytics directly; the
rest of the app talks to this ABC so the underlying model (YOLO, YOLO-NAS,
whatever comes next) can be swapped without touching callers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One raw prediction, in normalized xyxy of the ORIGINAL frame — the
    same contract `pipelines/frame_processor.py` in the sibling repo uses,
    so coordinates are pixel-exact-comparable across both codebases."""

    class_id: int
    class_name: str
    confidence: float
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
class Keypoint:
    """One COCO keypoint in normalized [0,1] image coordinates."""

    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class PoseResult:
    """One detected person: bbox (normalized xyxy) + 17 COCO keypoints,
    indexed 0-16 (ankles are 15 left, 16 right)."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    keypoints: list[Keypoint]


class DetectionModel(ABC):
    """Example concept from the spec, made concrete: `predict(image)` is the
    entire contract. An adapter for a different framework (YOLO-NAS, a
    future YOLO version) only has to satisfy this interface."""

    @property
    @abstractmethod
    def class_names(self) -> dict[int, str]:
        """The model's own class map, e.g. {0: 'ball', 1: 'cone', 2: 'cone_1'}
        — read from the weights, never hardcoded by a caller."""

    @abstractmethod
    def predict(
        self, image: np.ndarray, conf: float = 0.20, iou: float = 0.70, imgsz: int = 640
    ) -> list[Detection]:
        """Run detection on one BGR image (as loaded by cv2.imread), return
        raw detections — no filtering, no NMS beyond what iou= configures.
        Quality filtering is a separate layer (services/quality/), never
        baked into the model adapter."""

    def predict_batch(
        self, images: list[np.ndarray], conf: float = 0.20, iou: float = 0.70, imgsz: int = 640
    ) -> list[list[Detection]]:
        """Default: loop calling predict(). Adapters that support true
        batched inference (Ultralytics does) should override this for
        throughput — see UltralyticsDetectionModel.predict_batch."""
        return [self.predict(img, conf=conf, iou=iou, imgsz=imgsz) for img in images]

    def set_classes(self, classes: list[str]) -> None:
        """Re-embed the model's class prototypes from a text prompt list —
        only meaningful for an open-vocabulary model (YOLO-World). A
        closed-vocabulary adapter has a fixed taxonomy baked into its
        weights, so calling this on one is a caller bug: fail loudly rather
        than silently ignoring the requested classes."""
        raise NotImplementedError(f"{type(self).__name__} does not support dynamic classes")


class PoseModel(ABC):
    """Separate, optional interface for auxiliary keypoint models (e.g.
    pose_v1.pt). Not every project configures one — see
    services/quality/rules/packs/anatomical_proximity.py, the one consumer
    that requires it."""

    @abstractmethod
    def predict(self, image: np.ndarray, conf: float = 0.50, imgsz: int = 640) -> list[PoseResult]:
        ...


class ModelLoadError(RuntimeError):
    """Raised when a registered model file can't be loaded — surfaced to the
    API as a 4xx with a clear message, never silently swallowed into a
    "no predictions" result."""


def require_weights_file(path: Path) -> None:
    if not path.exists():
        raise ModelLoadError(f"Model weights not found at {path}")
