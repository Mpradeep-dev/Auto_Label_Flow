"""Database engine/session setup. `get_db` is the FastAPI dependency every
route uses; nothing outside `repositories/`/`services/` should import
`SessionLocal` directly.

Two dialects are supported from one codebase: PostgreSQL for the
docker-compose / server deployment, and a single-file SQLite database for
the standalone desktop app. SQLite needs a few connection pragmas (WAL so
the in-process job runner's writer doesn't lock out API reads, enforced
foreign keys, a busy timeout) that Postgres neither needs nor understands.
"""
from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

_is_sqlite = settings.DATABASE_URL.startswith("sqlite")

if _is_sqlite:
    engine = create_engine(
        settings.DATABASE_URL,
        future=True,
        # The in-process job runner (ThreadPoolExecutor) and the request
        # threadpool share this engine across threads.
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _connection_record) -> None:  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        # WAL: one writer + concurrent readers without "database is locked".
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        # Absorb brief writer-vs-writer contention (a job commit overlapping
        # a periodic reconcile/poll) instead of erroring immediately.
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()
else:
    engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True, future=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
