"""Modal connect/disconnect for the Settings page. Stores credentials in
the `integrations` table and mirrors them into the process environment +
`settings`, because the Modal SDK reads `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`
from `os.environ` directly — a DB row alone wouldn't reach it.
`load_on_startup` replays the stored row into the environment each time
the process boots, so a Docker restart doesn't lose the connection.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.integration import Integration, IntegrationProvider
from app.schemas.integration import IntegrationStatus

logger = logging.getLogger(__name__)


class ModalVerificationError(RuntimeError):
    pass


def apply_modal_env(token_id: str, token_secret: str) -> None:
    os.environ["MODAL_TOKEN_ID"] = token_id
    os.environ["MODAL_TOKEN_SECRET"] = token_secret
    settings.MODAL_TOKEN_ID = token_id
    settings.MODAL_TOKEN_SECRET = token_secret


def _verify(token_id: str, token_secret: str) -> None:
    """Verify Modal credentials by attempting to create a client."""
    apply_modal_env(token_id, token_secret)
    try:
        import modal

        client = modal.Client.from_token(token_id=token_id, token_secret=token_secret)
        # A lightweight call to verify the credentials work
        # modal.Client.profile() or similar — for now, just creating the
        # client is sufficient validation (it raises on bad credentials)
    except ImportError:
        raise ModalVerificationError("The `modal` package is not installed on this server")
    except Exception as exc:
        raise ModalVerificationError(f"Modal rejected these credentials: {exc}") from exc


def get_status(db: Session) -> IntegrationStatus:
    row = db.scalar(select(Integration).where(Integration.provider == IntegrationProvider.MODAL.value))
    if row is None:
        return IntegrationStatus(provider="MODAL", connected=False)
    return IntegrationStatus(
        provider="MODAL",
        connected=row.verified_at is not None,
        identifier=row.config.get("token_id", "")[:12] + "..." if row.config.get("token_id") else None,
        verified_at=row.verified_at,
        last_error=row.last_error,
    )


def connect(db: Session, *, token_id: str, token_secret: str) -> IntegrationStatus:
    row = db.scalar(select(Integration).where(Integration.provider == IntegrationProvider.MODAL.value))
    if row is None:
        row = Integration(provider=IntegrationProvider.MODAL.value, config={})
        db.add(row)

    try:
        _verify(token_id, token_secret)
    except Exception as exc:
        logger.exception("Modal connect failed")
        row.config = {"token_id": token_id, "token_secret": token_secret}
        row.verified_at = None
        row.last_error = str(exc)
        db.commit()
        raise

    row.config = {"token_id": token_id, "token_secret": token_secret}
    row.verified_at = datetime.now(timezone.utc)
    row.last_error = None
    db.commit()
    db.refresh(row)
    return get_status(db)


def disconnect(db: Session) -> None:
    row = db.scalar(select(Integration).where(Integration.provider == IntegrationProvider.MODAL.value))
    if row is not None:
        db.delete(row)
        db.commit()
    os.environ.pop("MODAL_TOKEN_ID", None)
    os.environ.pop("MODAL_TOKEN_SECRET", None)
    settings.MODAL_TOKEN_ID = None
    settings.MODAL_TOKEN_SECRET = None


def load_on_startup(db: Session) -> None:
    """Called once at app startup — replays a previously connected Modal
    account into the environment so credentials survive a process restart
    without the user reconnecting."""
    row = db.scalar(select(Integration).where(Integration.provider == IntegrationProvider.MODAL.value))
    if row is not None and row.verified_at is not None and row.config.get("token_id") and row.config.get("token_secret"):
        apply_modal_env(row.config["token_id"], row.config["token_secret"])
