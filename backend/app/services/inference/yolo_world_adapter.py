"""YOLO-World adapter — an open-vocabulary detector whose class prototypes
are re-embedded from a text prompt list at inference time via
`set_classes(...)`, instead of being fixed by the weights at load time like
every other `DetectionModel`. See `services/inference/registry.py` for how
the prompt list (a project's `class_config` names) gets threaded in.

Constructs `ultralytics.YOLOWorld` explicitly rather than relying on
`ultralytics.YOLO(path)`'s filename-sniffing dispatch (which only picks
YOLOWorld up if the weights filename stem happens to contain "-world") —
this repo renames uploaded/downloaded weights to `slugify(name)-<hash>.pt`,
so that dispatch is not reliable here.
"""
from __future__ import annotations

from pathlib import Path

from app.services.inference.detector import ModelLoadError, require_weights_file
from app.services.inference.ultralytics_adapter import UltralyticsDetectionModel, _resolve_device


class YoloWorldDetectionModel(UltralyticsDetectionModel):
    def __init__(self, weights_path: str) -> None:  # noqa: super-init-not-called (deliberately reimplemented)
        require_weights_file(Path(weights_path))
        from ultralytics import YOLOWorld

        try:
            self._model = YOLOWorld(weights_path)
        except Exception as exc:  # pragma: no cover - depends on corrupt weights, hard to fixture
            raise ModelLoadError(f"Failed to load YOLO-World weights at {weights_path}: {exc}") from exc

        device = _resolve_device()
        try:
            self._model.overrides["device"] = device
        except Exception:
            self._model.overrides["device"] = "cpu"  # last-resort fallback, never crash on load

        # `model.names` at this point is whatever default vocabulary the
        # checkpoint ships with (usually COCO's 80 classes) — a display
        # fallback only, never authoritative; the real vocabulary is
        # whatever set_classes() is called with before predict.
        self._class_names: dict[int, str] = {int(k): v for k, v in self._model.names.items()}

    @property
    def class_names(self) -> dict[int, str]:
        return dict(self._class_names)

    def set_classes(self, classes: list[str]) -> None:
        if not classes:
            raise ValueError("YOLO-World requires a non-empty class list to detect anything")
        self._model.set_classes(classes)
        # Ultralytics mutates `model.names` to a plain list after
        # set_classes() — normalize back to the dict[int, str] contract the
        # rest of this codebase (predict_batch, class_names) expects.
        self._class_names = dict(enumerate(classes))
