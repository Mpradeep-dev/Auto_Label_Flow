"""SQLite schema bootstrap for the standalone desktop app.

The server deployment runs Alembic (`alembic upgrade head`) against
PostgreSQL. The desktop app ships a single-file SQLite database that no
user has ever run before, so there is nothing to *migrate from* — and
several of the Alembic migrations carry Postgres-only native-enum DDL that
SQLite cannot execute anyway. Instead, the schema is materialised once,
straight from `Base.metadata`, and stamped with a version in
`PRAGMA user_version`.

Future desktop schema changes ship as small idempotent steps keyed off
`user_version` here; there are none yet.

This is the same `create_all` path the test suite uses, so it is covered by
every backend test that runs against SQLite.
"""
from __future__ import annotations

import logging

from sqlalchemy import Engine, text

import app.models  # noqa: F401  — imports every model so Base.metadata is complete
from app.db.base import Base

logger = logging.getLogger(__name__)

# Bump when a released desktop build changes the schema, and add the
# corresponding upgrade step in `_upgrade()` below.
SCHEMA_VERSION = 1


def init_sqlite_schema(engine: Engine) -> None:
    """Create the schema on first run; no-op once stamped. Safe to call on
    every startup. Only touches SQLite engines."""
    if engine.dialect.name != "sqlite":
        return

    with engine.begin() as conn:
        current = conn.exec_driver_sql("PRAGMA user_version").scalar_one()

        if current == 0:
            logger.info("initialising SQLite schema at version %s", SCHEMA_VERSION)
            Base.metadata.create_all(conn)
            conn.exec_driver_sql(f"PRAGMA user_version = {SCHEMA_VERSION}")
            return

        if current < SCHEMA_VERSION:
            logger.info("upgrading SQLite schema %s -> %s", current, SCHEMA_VERSION)
            _upgrade(conn, from_version=current)
            conn.exec_driver_sql(f"PRAGMA user_version = {SCHEMA_VERSION}")
        elif current > SCHEMA_VERSION:
            logger.warning(
                "SQLite schema version %s is newer than this build's %s "
                "(a downgrade?) — leaving it alone",
                current,
                SCHEMA_VERSION,
            )


def _upgrade(conn, from_version: int) -> None:  # noqa: ANN001
    """Ordered, idempotent schema steps for the desktop SQLite DB. Empty for
    now — every released schema so far is `SCHEMA_VERSION == 1`."""
    _ = (conn, from_version, text)  # placeholder until the first real step
