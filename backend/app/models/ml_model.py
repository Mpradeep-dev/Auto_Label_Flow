"""ML model registry. A `Model` row is global (not project-scoped) — one
registered model can serve as the primary detector for several projects, or
be swapped in after retraining. Projects reference one via
`Project.primary_model_id` / `Project.pose_model_id`.

`class_config` is read from the weights (`model.names`) at registration
time via `services/inference/registry.py::register_model` and copied here
so a project's active taxonomy doesn't have to re-load the weights on every
read — see PLAN Decision 1."""
from __future__ import annotations

import uuid
from enum import Enum as PyEnum

from sqlalchemy import Boolean, Enum, ForeignKey, JSON, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class ModelKind(str, PyEnum):
    DETECTOR = "DETECTOR"
    POSE = "POSE"
    # SAM/SAM2 segmentation support (a SEGMENTER kind) was removed — the
    # Postgres `model_kind` enum still carries the value from migration
    # 9f2b6d4e8a17 (ALTER TYPE ... ADD VALUE can't be undone), but nothing
    # in the app offers or reads it anymore; no row uses it.


class MLModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "models"
    # Audit finding DB-03: registering the same name+version twice used to
    # silently create two confusing rows in the model picker. Safe to
    # enforce now that the training-completion auto-registration path
    # (workers/tasks/training.py) gives each retrain of the same base model
    # a distinct version string instead of a fixed "trained-from-<name>" —
    # before that fix, retraining from the same base twice would have hit
    # this constraint on a perfectly legitimate second retrain.
    __table_args__ = (UniqueConstraint("name", "version", name="uq_models_name_version"),)

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False, default="v1")
    kind: Mapped[ModelKind] = mapped_column(Enum(ModelKind, name="model_kind"), nullable=False)
    framework: Mapped[str] = mapped_column(String(50), nullable=False, default="ultralytics")

    # Absolute filesystem path to the weights file (under ARTIFACTS_DIR).
    # Not an ObjectStorage key: model weights are a deploy-time artifact
    # mounted into every backend/worker container identically (see
    # docker-compose.yml `artifacts:` volume), not per-upload user content.
    weights_path: Mapped[str] = mapped_column(String(1000), nullable=False)

    # [{"id": 0, "name": "ball"}, ...] — read from model.names at
    # registration time (services/inference/registry.py), never typed in.
    class_config: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    # Populated by evaluation upload (Phase 8) / training completion (Phase 6):
    # {"map50": ..., "map50_95": ..., "precision": ..., "recall": ...,
    #  "per_class": {"cone": {"precision": ..., "recall": ...}, ...}, "fps": ...}
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    # Training lineage: the model this one was fine-tuned from, if any.
    # A real FK (audit finding DB-02) — was a bare UUID column with no
    # constraint, inconsistent with TrainingJob.base_model_id/result_model_id,
    # which already reference this same table. SET NULL, not RESTRICT/CASCADE:
    # deleting a base model shouldn't be blocked by, or take down, models
    # that were fine-tuned from it — it should just drop the now-stale
    # lineage pointer, same as TrainingJob's.
    base_model_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("models.id", ondelete="SET NULL"), nullable=True
    )

    is_active: Mapped[bool] = mapped_column(nullable=False, default=True)

    # True for an open-vocabulary detector (YOLO-World): `class_config`
    # above is only a display fallback (the checkpoint's default
    # vocabulary), never authoritative — the real classes are supplied at
    # inference time via DetectionModel.set_classes(), see
    # services/inference/registry.py::get_detection_model.
    is_promptable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
