"""Cross-dialect column types.

The same ORM models back two databases now: PostgreSQL for the
docker-compose / server deployment, and a bundled **SQLite** file for the
standalone desktop app. These `TypeDecorator`s keep the Postgres DDL
byte-identical to what it has always been while letting the exact same
models build a working schema on SQLite via `Base.metadata.create_all()`.

  - `GUID`        — `uuid.UUID` <-> native PG `UUID` / `CHAR(32)` hex on SQLite
  - `TZDateTime`  — tz-aware datetime that survives SQLite's tz-naive storage
  - `enum_column` — native PG `ENUM`, `VARCHAR + CHECK` on SQLite

Nothing dialect-specific should be imported into `app/models/` directly any
more — go through here.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.types import CHAR, DateTime, TypeDecorator


class GUID(TypeDecorator):
    """Platform-independent UUID column.

    On PostgreSQL this is the native `UUID` type with `as_uuid=True`
    semantics, so every repository / service / schema that already passes
    and receives `uuid.UUID` is unaffected. On any other dialect the value
    is stored as a 32-char hex string (no dashes — keeps the index small).
    Binds and returns `uuid.UUID` on every dialect.
    """

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):  # noqa: ANN001
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PGUUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(32))

    def process_bind_param(self, value, dialect):  # noqa: ANN001
        if value is None:
            return None
        if not isinstance(value, uuid.UUID):
            value = uuid.UUID(str(value))
        if dialect.name == "postgresql":
            return value
        return value.hex

    def process_result_value(self, value, dialect):  # noqa: ANN001
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(str(value))


class TZDateTime(TypeDecorator):
    """Timezone-aware datetime that keeps working on SQLite.

    SQLite has no tz-aware storage, so SQLAlchemy hands back **naive**
    datetimes from a `DateTime(timezone=True)` column there. Code that then
    compares them against `datetime.now(timezone.utc)` (e.g.
    `workers/tasks/reconcile.py`) raises `TypeError: can't compare
    offset-naive and offset-aware datetimes`. This decorator normalises to
    UTC and drops tzinfo on the way in, and re-attaches UTC on the way out,
    for non-Postgres dialects. On Postgres it is a straight pass-through to
    `DateTime(timezone=True)`.
    """

    impl = DateTime
    cache_ok = True

    def load_dialect_impl(self, dialect):  # noqa: ANN001
        return dialect.type_descriptor(DateTime(timezone=True))

    def process_bind_param(self, value, dialect):  # noqa: ANN001
        if value is None or dialect.name == "postgresql":
            return value
        if isinstance(value, datetime) and value.tzinfo is not None:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value

    def process_result_value(self, value, dialect):  # noqa: ANN001
        if value is None or dialect.name == "postgresql":
            return value
        if isinstance(value, datetime) and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


def enum_column(py_enum: type, name: str):
    """A `sa.Enum` that stays a native `CREATE TYPE` enum on PostgreSQL —
    byte-identical DDL to before, so Alembic autogenerate sees no diff — but
    degrades to `VARCHAR + CHECK` on SQLite, which has neither native enums
    nor `ALTER TYPE ... ADD VALUE`. Use this instead of a bare
    `native_enum=False`, which *would* be a breaking Postgres schema change.
    """
    return sa.Enum(py_enum, name=name).with_variant(
        sa.Enum(py_enum, name=name, native_enum=False),
        "sqlite",
    )
