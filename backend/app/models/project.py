"""Project — the top-level container. `class_config` is the project's active
class list: seeded from whichever model is registered as its primary
detector (`model.names`, read at registration time — see
`services/inference/registry.py`) and editable from there. Nothing in this
schema hardcodes a taxonomy; a project loaded against a completely different
model just gets a different `class_config`."""
from __future__ import annotations

import uuid

from sqlalchemy import JSON, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Project(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "projects"

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(200), nullable=False, unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # [{"id": 0, "name": "ball"}, {"id": 1, "name": "cone"}, ...] — read from
    # the primary detector's own weights at registration time, never typed
    # in by hand. Colour is NOT stored here: the frontend derives colour
    # deterministically from list position (see PLAN "Box colour language").
    class_config: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    # Per-project overrides for pluggable quality rules, e.g.
    # {"anatomical_proximity": {"enabled": true, "target_class_ids": [1, 2]}}.
    # Empty dict = every class-agnostic rule runs with its defaults, no
    # optional packs active.
    quality_rule_config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    primary_model_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("models.id", ondelete="SET NULL"), nullable=True
    )
    pose_model_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("models.id", ondelete="SET NULL"), nullable=True
    )

    datasets: Mapped[list["Dataset"]] = relationship(back_populates="project", cascade="all, delete-orphan")
