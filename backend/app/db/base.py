"""Declarative base + shared column mixins. Every ORM model in `app/models/`
inherits `Base`, and most inherit `TimestampMixin`/`UUIDPrimaryKeyMixin` too,
so id/timestamp conventions live in exactly one place.

Column types come from `app.db.types` (`GUID`, `TZDateTime`) so the same
models build on both PostgreSQL and the desktop app's bundled SQLite."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.db.types import GUID, TZDateTime


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class UUIDPrimaryKeyMixin:
    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(TZDateTime, default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        TZDateTime, default=_utcnow, onupdate=_utcnow, nullable=False
    )
