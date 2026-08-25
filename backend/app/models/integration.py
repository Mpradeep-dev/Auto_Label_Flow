"""Third-party service connections (Kaggle, Roboflow) — application-wide,
not per-project (PLAN "Credential scope": one person's account either way,
matches how Kaggle already worked via env vars). One row per provider,
upserted on (re)connect. `config` holds whatever that provider's client
needs (Kaggle: username+key; Roboflow: api_key+default_workspace) — no
column-per-field so a new provider is a new row shape, not a migration.

Credentials are stored in plaintext, same trust boundary as the existing
`backend/.env` file (KAGGLE_KEY has always lived there in cleartext) — this
app has no auth layer (see core/config.py), so encrypting at rest would
need a key stored exactly as unprotected as the thing it's meant to guard.
The API layer (schemas/integration.py) masks secrets before they're ever
read back to the frontend.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import DateTime, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class IntegrationProvider(str, PyEnum):
    KAGGLE = "KAGGLE"
    ROBOFLOW = "ROBOFLOW"


class Integration(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "integrations"

    provider: Mapped[str] = mapped_column(String(20), nullable=False, unique=True, index=True)
    config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
