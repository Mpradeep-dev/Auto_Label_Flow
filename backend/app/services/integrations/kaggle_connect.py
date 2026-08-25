"""Kaggle connect/disconnect for the Settings page. Stores credentials in
the `integrations` table and mirrors them into the process environment +
`settings`, because the `kaggle` package's `KaggleApi.authenticate()`
reads `KAGGLE_USERNAME`/`KAGGLE_KEY` from `os.environ` directly (confirmed
in `services/training/kaggle_provider.py`) — a DB row alone wouldn't reach
it. `load_on_startup` replays the stored row into the environment each
time the process boots, so a Docker restart doesn't lose the connection.
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


class KaggleVerificationError(RuntimeError):
    pass


def apply_kaggle_env(username: str, key: str) -> None:
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key
    settings.KAGGLE_USERNAME = username
    settings.KAGGLE_KEY = key


def _verify(username: str, key: str) -> None:
    """A real network call, not just "did authenticate() not throw" — that
    method only checks the two env vars are non-empty.

    `user_show` doesn't exist on the pinned `kaggle==1.6.17` client — that
    always failed with AttributeError regardless of credentials.
    `competitions_submissions_list("titanic")` was tried too, but it
    rejects perfectly valid credentials with 400 "You do not have a Team
    in this Competition" whenever the account simply hasn't joined that
    competition — a false negative unrelated to key validity.
    `kernels_list(mine=True)` (confirmed live: 401 "Unauthorized" for
    fabricated credentials) enforces auth without depending on membership
    in any competition/dataset the account may not have touched."""
    apply_kaggle_env(username, key)
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kaggle.rest import ApiException

    api = KaggleApi()
    api.authenticate()
    try:
        api.kernels_list(mine=True)
    except ApiException as exc:
        # str(ApiException) dumps the entire response including every HTTP
        # header — status + reason is what a user actually needs to see.
        raise KaggleVerificationError(f"Kaggle rejected these credentials: {exc.status} {exc.reason}") from exc
    except Exception as exc:
        raise KaggleVerificationError(f"Could not reach Kaggle: {exc}") from exc


def get_status(db: Session) -> IntegrationStatus:
    row = db.scalar(select(Integration).where(Integration.provider == IntegrationProvider.KAGGLE.value))
    if row is None:
        return IntegrationStatus(provider="KAGGLE", connected=False)
    # A row can exist with verified_at=None: a failed connect attempt still
    # persists the entered config (so the user doesn't retype it after
    # fixing a typo), but that is NOT "connected" — only a successful
    # verification sets verified_at.
    return IntegrationStatus(
        provider="KAGGLE",
        connected=row.verified_at is not None,
        identifier=row.config.get("username"),
        verified_at=row.verified_at,
        last_error=row.last_error,
    )


def connect(db: Session, *, username: str, key: str) -> IntegrationStatus:
    row = db.scalar(select(Integration).where(Integration.provider == IntegrationProvider.KAGGLE.value))
    if row is None:
        row = Integration(provider=IntegrationProvider.KAGGLE.value, config={})
        db.add(row)

    try:
        _verify(username, key)
    except Exception as exc:
        # _verify() already converts everything it can anticipate into
        # KaggleVerificationError; logging here catches the case where
        # something outside that (a truly unexpected exception) would
        # otherwise surface to the caller as an unlogged 500.
        logger.exception("Kaggle connect failed")
        row.config = {"username": username, "key": key}
        row.verified_at = None
        row.last_error = str(exc)
        db.commit()
        raise

    row.config = {"username": username, "key": key}
    row.verified_at = datetime.now(timezone.utc)
    row.last_error = None
    db.commit()
    db.refresh(row)
    return get_status(db)


def disconnect(db: Session) -> None:
    row = db.scalar(select(Integration).where(Integration.provider == IntegrationProvider.KAGGLE.value))
    if row is not None:
        db.delete(row)
        db.commit()
    os.environ.pop("KAGGLE_USERNAME", None)
    os.environ.pop("KAGGLE_KEY", None)
    settings.KAGGLE_USERNAME = None
    settings.KAGGLE_KEY = None


def load_on_startup(db: Session) -> None:
    """Called once at app startup (see `main.py`) — replays a previously
    connected Kaggle account into the environment so credentials survive a
    process restart without the user reconnecting."""
    row = db.scalar(select(Integration).where(Integration.provider == IntegrationProvider.KAGGLE.value))
    if row is not None and row.verified_at is not None and row.config.get("username") and row.config.get("key"):
        apply_kaggle_env(row.config["username"], row.config["key"])
