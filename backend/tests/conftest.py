"""Shared pytest fixtures.

House convention borrowed from the sibling gsp-video-ai-processing-service
repo: env stubs are set *before* any app import, since Settings validate at
import time. Here that means pointing DATABASE_URL at the dedicated
`annotate_test` database before `app.core.config` (or anything importing
it) is ever touched.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from collections.abc import Generator
from pathlib import Path

_TEST_STORAGE_DIR = Path(tempfile.gettempdir()) / "annotate_test_storage"

os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg://annotate:annotate@localhost:5432/annotate_test"
)
os.environ.setdefault("STORAGE_BACKEND", "local")
os.environ.setdefault("LOCAL_STORAGE_DIR", str(_TEST_STORAGE_DIR))
os.environ.setdefault("ENV", "test")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base
from app.db.session import get_db
from app.main import app

_engine = create_engine(os.environ["DATABASE_URL"], future=True)
_TestSessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)


@pytest.fixture(scope="session", autouse=True)
def _create_schema() -> Generator[None, None, None]:
    _TEST_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(_engine)
    yield
    Base.metadata.drop_all(_engine)
    shutil.rmtree(_TEST_STORAGE_DIR, ignore_errors=True)


@pytest.fixture()
def db_session() -> Generator[Session, None, None]:
    """Each test gets a fresh, empty set of tables' worth of *rows* via a
    transaction rolled back on teardown — schema is created once per
    session (see `_create_schema`), not per test, since DDL is comparatively
    expensive and immutable within a test run."""
    connection = _engine.connect()
    transaction = connection.begin()
    session = _TestSessionLocal(bind=connection)
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture()
def client(db_session: Session) -> Generator[TestClient, None, None]:
    def _override_get_db() -> Generator[Session, None, None]:
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture()
def unique_name() -> str:
    return f"test-{uuid.uuid4().hex[:8]}"


@pytest.fixture()
def real_db_session() -> Generator[Session, None, None]:
    """A genuinely-committing session, distinct from `db_session`'s
    rollback-on-teardown transaction wrapper. Needed for tests that
    exercise Celery tasks run in eager mode: a task opens its OWN session
    via `SessionLocal()` (a separate connection), which would never see
    data written through `db_session`'s uncommitted outer transaction.
    Test data written here isn't rolled back per-test, but the whole
    `annotate_test` schema is dropped at session end (`_create_schema`),
    and callers use uuid-suffixed names, so cross-test collisions aren't a
    concern."""
    session = _TestSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def real_client(real_db_session: Session) -> Generator[TestClient, None, None]:
    def _override_get_db() -> Generator[Session, None, None]:
        yield real_db_session

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
