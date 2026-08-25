"""Roboflow connect/disconnect for the Settings page, and the shared client
factory `get_client()` used by both `roboflow_export.py` and
`roboflow_import.py`. Unlike Kaggle, the `roboflow` SDK takes its API key
directly as a constructor argument rather than reading the environment, so
there's no env-mirroring step here — just read the stored key and build a
client per call.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.integration import Integration, IntegrationProvider
from app.schemas.integration import IntegrationStatus

logger = logging.getLogger(__name__)


class RoboflowNotConnectedError(RuntimeError):
    pass


class RoboflowVerificationError(RuntimeError):
    pass


class RoboflowDestructiveOperationBlocked(RuntimeError):
    """Raised in place of ever letting a delete/remove call reach the
    Roboflow API. This integration is upload/import only by design (PLAN
    "never delete or modify data in a connected Roboflow account") —
    `_install_destructive_guards` enforces that at the SDK class level so
    the guarantee holds regardless of which service module calls it, this
    one or any added later."""


_destructive_guards_installed = False

# Every method on these SDK classes that can delete or remove something
# server-side. Everything else on them (upload, download, search, read) is
# left alone — this blocks writes-that-destroy, not writes-that-create.
_DESTRUCTIVE_METHODS: dict[str, tuple[str, ...]] = {
    "project": ("delete", "delete_images", "delete_annotation_batch", "delete_annotation_job_annotations"),
    "workspace": ("delete_images", "remove_projects_from_folder"),
    "version": ("delete", "delete_training"),
    "training": ("delete",),
}


def _install_destructive_guards() -> None:
    """Monkeypatches the `roboflow` SDK's own classes so every delete-ish
    method raises instead of hitting the network — a single, permanent,
    process-wide guard installed the first time this module ever touches
    the SDK (both `connect()` and `get_client()` call this), rather than
    something each call site has to remember to avoid."""
    global _destructive_guards_installed
    if _destructive_guards_installed:
        return

    from roboflow.core.project import Project
    from roboflow.core.training import Training
    from roboflow.core.version import Version
    from roboflow.core.workspace import Workspace

    def _make_blocker(qualname: str):
        def _blocked(self, *args, **kwargs):
            raise RoboflowDestructiveOperationBlocked(
                f"Blocked: this app called Roboflow SDK's {qualname}(), which deletes/removes data in your "
                "connected Roboflow account. This integration is import/export only and must never do that — "
                "the call was refused before it reached the network."
            )

        return _blocked

    for cls in (Project, Workspace, Version, Training):
        for method_name in _DESTRUCTIVE_METHODS[cls.__name__.lower()]:
            if hasattr(cls, method_name):
                setattr(cls, method_name, _make_blocker(f"{cls.__name__}.{method_name}"))

    _destructive_guards_installed = True


def get_status(db: Session) -> IntegrationStatus:
    row = db.scalar(select(Integration).where(Integration.provider == IntegrationProvider.ROBOFLOW.value))
    if row is None:
        return IntegrationStatus(provider="ROBOFLOW", connected=False)
    # See kaggle_connect.get_status: a row can exist from a failed attempt
    # (verified_at still None) — that isn't "connected".
    return IntegrationStatus(
        provider="ROBOFLOW",
        connected=row.verified_at is not None,
        identifier=row.config.get("default_workspace"),
        verified_at=row.verified_at,
        last_error=row.last_error,
    )


def connect(db: Session, *, api_key: str, default_workspace: str | None) -> IntegrationStatus:
    # Imported lazily (mirrors kaggle_provider.py): keeps the `roboflow`
    # package optional for installs that never use this integration.
    from roboflow import Roboflow

    _install_destructive_guards()

    row = db.scalar(select(Integration).where(Integration.provider == IntegrationProvider.ROBOFLOW.value))
    if row is None:
        row = Integration(provider=IntegrationProvider.ROBOFLOW.value, config={})
        db.add(row)

    try:
        # Roboflow(api_key=...) itself calls check_key() against the API
        # and raises on an invalid key (traced via roboflow.Roboflow.auth) —
        # this IS the verification, not a formality before one.
        #
        # One real gap in Roboflow's own check_key(): if the key string has
        # no lowercase characters at all, it skips the network call
        # entirely and returns the sentinel "onboarding" — auth() then
        # never sets current_workspace and treats it as success. A key
        # shaped that way (garbage, or copy-paste-mangled to uppercase)
        # would otherwise "connect" without ever having been checked.
        rf = Roboflow(api_key=api_key)
        if getattr(rf, "onboarding", False):
            raise RuntimeError(
                "Roboflow could not validate this key (it never reached the API — the key format wasn't "
                "recognized). Double-check it was copied correctly."
            )
        resolved_workspace = default_workspace or rf.current_workspace
    except Exception as exc:
        # Deliberately broad: anything that goes wrong here — a bad key, a
        # network hiccup, an unexpected response shape from Roboflow's own
        # SDK — should come back as a clear message the user can act on,
        # never an opaque 500. logger.exception keeps the real traceback in
        # server logs for cases genuinely worth debugging (this endpoint's
        # own except clauses only recognize specific error types, so
        # anything else raised here would otherwise surface to the caller
        # as an unlogged, unexplained Internal Server Error).
        logger.exception("Roboflow connect failed")
        row.config = {"api_key": api_key, "default_workspace": default_workspace}
        row.verified_at = None
        row.last_error = str(exc)
        db.commit()
        raise RoboflowVerificationError(f"Roboflow rejected this API key: {exc}") from exc

    row.config = {"api_key": api_key, "default_workspace": resolved_workspace}
    row.verified_at = datetime.now(timezone.utc)
    row.last_error = None
    db.commit()
    db.refresh(row)
    return get_status(db)


def disconnect(db: Session) -> None:
    row = db.scalar(select(Integration).where(Integration.provider == IntegrationProvider.ROBOFLOW.value))
    if row is not None:
        db.delete(row)
        db.commit()


def get_client(db: Session):
    """Returns (Roboflow client, stored config dict). Raises
    RoboflowNotConnectedError if Settings hasn't connected an account yet —
    callers turn that into a 400, same pattern as
    `KaggleNotConfiguredError` in kaggle_provider.py.

    Every caller in this codebase gets its `Project`/`Workspace`/`Version`
    objects by calling `.workspace()`/`.project()`/`.version()` off the
    client this returns — which makes this the one place guaranteed to run
    before any of those objects exist, so `_install_destructive_guards()`
    here is enough to cover the whole app, present and future."""
    from roboflow import Roboflow

    _install_destructive_guards()

    row = db.scalar(select(Integration).where(Integration.provider == IntegrationProvider.ROBOFLOW.value))
    if row is None or row.verified_at is None or not row.config.get("api_key"):
        raise RoboflowNotConnectedError("Roboflow is not connected — add an API key in Settings first.")
    return Roboflow(api_key=row.config["api_key"]), row.config
