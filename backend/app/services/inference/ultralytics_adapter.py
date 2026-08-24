"""The reference DetectionModel/PoseModel implementation, backed by
Ultralytics YOLO. This is the ONLY module that imports `ultralytics` for
inference — everything else in the app talks to the `DetectionModel`/
`PoseModel` ABCs, so a YOLO-NAS or other adapter can be dropped in later
without touching callers (PLAN: "Do not tightly couple the application to
one specific detector.").

Inference call pattern (letterbox -> model(..., conf=, iou=, imgsz=) ->
boxes_to_original) mirrors gsp-video-ai-processing-service
`pipelines/frame_processor.py` exactly, so this platform's coordinate frame
and confidence numbers are directly comparable to production's.
"""
from __future__ import annotations

import numpy as np

from app.services.inference.detector import (
    DetectionModel,
    Detection,
    Keypoint,
    ModelLoadError,
    PoseModel,
    PoseResult,
    require_weights_file,
)
from app.services.inference.letterbox import boxes_to_original, keypoints_to_original, letterbox
from pathlib import Path


def _resolve_device() -> str:
    """cuda:0 if a GPU is visible and usable, else cpu — mirrors the
    sibling repo's `_cuda_device()` probe (infrastructure/model_runtime/models.py).
    Deliberately simpler than that module's full TensorRT/OpenVINO/ONNX/
    PyTorch chain: this registry stores one weights file per Model row
    (see app/models/ml_model.py), not the sibling's pre-exported
    multi-format artifact set, so there is only one format to load — the
    thing worth falling back on here is the *device*, not the file format."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:
        pass
    return "cpu"


class UltralyticsDetectionModel(DetectionModel):
    def __init__(self, weights_path: str) -> None:
        require_weights_file(Path(weights_path))
        from ultralytics import YOLO

        try:
            self._model = YOLO(weights_path)
        except Exception as exc:  # pragma: no cover - depends on corrupt weights, hard to fixture
            raise ModelLoadError(f"Failed to load detector weights at {weights_path}: {exc}") from exc

        device = _resolve_device()
        try:
            self._model.overrides["device"] = device
        except Exception:
            self._model.overrides["device"] = "cpu"  # last-resort fallback, never crash on load

        self._class_names: dict[int, str] = {int(k): v for k, v in self._model.names.items()}

    @property
    def class_names(self) -> dict[int, str]:
        return dict(self._class_names)

    def predict(
        self, image: np.ndarray, conf: float = 0.20, iou: float = 0.70, imgsz: int = 640
    ) -> list[Detection]:
        return self.predict_batch([image], conf=conf, iou=iou, imgsz=imgsz)[0]

    def predict_batch(
        self, images: list[np.ndarray], conf: float = 0.20, iou: float = 0.70, imgsz: int = 640
    ) -> list[list[Detection]]:
        if not images:
            return []

        letterboxed = []
        metas = []
        for img in images:
            lb, meta = letterbox(img, (imgsz, imgsz))
            letterboxed.append(lb)
            metas.append(meta)

        results = self._model(letterboxed, verbose=False, conf=conf, iou=iou, imgsz=imgsz)

        out: list[list[Detection]] = []
        for result, meta in zip(results, metas):
            detections: list[Detection] = []
            if result.boxes is not None and len(result.boxes) > 0:
                boxes_orig = boxes_to_original(result.boxes.xyxy.cpu().numpy(), meta)
                cls_arr = result.boxes.cls.cpu().numpy().astype(int)
                conf_arr = result.boxes.conf.cpu().numpy()
                for bbox, cls_id, score in zip(boxes_orig, cls_arr, conf_arr):
                    x1, y1, x2, y2 = bbox
                    detections.append(
                        Detection(
                            class_id=int(cls_id),
                            class_name=self._class_names.get(int(cls_id), str(cls_id)),
                            confidence=float(score),
                            x1=float(x1 / meta.orig_width),
                            y1=float(y1 / meta.orig_height),
                            x2=float(x2 / meta.orig_width),
                            y2=float(y2 / meta.orig_height),
                        )
                    )
            out.append(detections)
        return out


class UltralyticsPoseModel(PoseModel):
    def __init__(self, weights_path: str) -> None:
        require_weights_file(Path(weights_path))
        from ultralytics import YOLO

        try:
            self._model = YOLO(weights_path)
        except Exception as exc:  # pragma: no cover
            raise ModelLoadError(f"Failed to load pose weights at {weights_path}: {exc}") from exc

        device = _resolve_device()
        try:
            self._model.overrides["device"] = device
        except Exception:
            self._model.overrides["device"] = "cpu"

    def predict(self, image: np.ndarray, conf: float = 0.50, imgsz: int = 640) -> list[PoseResult]:
        lb, meta = letterbox(image, (imgsz, imgsz))
        results = self._model(lb, verbose=False, conf=conf, imgsz=imgsz)
        result = results[0]

        people: list[PoseResult] = []
        if result.boxes is None or len(result.boxes) == 0:
            return people

        boxes_orig = boxes_to_original(result.boxes.xyxy.cpu().numpy(), meta)
        box_confs = result.boxes.conf.cpu().numpy()

        has_kpts = result.keypoints is not None and len(result.keypoints) > 0
        for i, (bbox, box_conf) in enumerate(zip(boxes_orig, box_confs)):
            x1, y1, x2, y2 = bbox
            keypoints: list[Keypoint] = []
            if has_kpts:
                kxy = keypoints_to_original(result.keypoints.xy[i].cpu().numpy(), meta)
                kconf = (
                    result.keypoints.conf[i].cpu().numpy()
                    if result.keypoints.conf is not None
                    else np.ones(len(kxy))
                )
                for (kx, ky), kc in zip(kxy, kconf):
                    keypoints.append(
                        Keypoint(
                            x=float(kx / meta.orig_width),
                            y=float(ky / meta.orig_height),
                            confidence=float(kc),
                        )
                    )
            people.append(
                PoseResult(
                    x1=float(x1 / meta.orig_width),
                    y1=float(y1 / meta.orig_height),
                    x2=float(x2 / meta.orig_width),
                    y2=float(y2 / meta.orig_height),
                    confidence=float(box_conf),
                    keypoints=keypoints,
                )
            )
        return people
