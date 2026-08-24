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

from sqlalchemy import Enum, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class ModelKind(str, PyEnum):
    DETECTOR = "DETECTOR"
    POSE = "POSE"


class MLModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "models"

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
    base_model_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)

    is_active: Mapped[bool] = mapped_column(nullable=False, default=True)
