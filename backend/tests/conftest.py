"""Shared pytest fixtures.

House convention borrowed from the sibling gsp-video-ai-processing-service
repo: env stubs are set *before* any app import, since Settings validate at
import time. Here that means pointing DATABASE_URL at a throwaway database
before `app.core.config` (or anything importing it) is ever touched.

Default target is a temp-file **SQLite** database — the same engine the
desktop app ships, exercising the real WAL / foreign-keys / busy-timeout
pragmas and the `create_all` schema path. Set `DATABASE_URL` explicitly
(e.g. to `postgresql+psycopg://...annotate_test`) to run the suite against
the server stack instead; the CI matrix does both.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from collections.abc import Generator
from pathlib import Path

_TEST_STORAGE_DIR = Path(tempfile.gettempdir()) / "annotate_test_storage"
# A file, not ":memory:" — conftest opens several independent connections
# (real_db_session, eager tasks) that an in-memory DB would not share.
_TEST_DB_FILE = Path(tempfile.gettempdir()) / "annotate_test.db"

os.environ.setdefault("DATABASE_URL", f"sqlite+pysqlite:///{_TEST_DB_FILE.as_posix()}")
os.environ.setdefault("STORAGE_BACKEND", "local")
os.environ.setdefault("LOCAL_STORAGE_DIR", str(_TEST_STORAGE_DIR))
os.environ.setdefault("ENV", "test")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base
from app.db.session import get_db
from app.main import app

_DB_URL = os.environ["DATABASE_URL"]
_IS_SQLITE = _DB_URL.startswith("sqlite")
_engine = create_engine(
    _DB_URL,
    future=True,
    connect_args={"check_same_thread": False} if _IS_SQLITE else {},
)

if _IS_SQLITE:

    @event.listens_for(_engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _record) -> None:  # noqa: ANN001
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.close()

_TestSessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)


def _rm_sqlite_files() -> None:
    for suffix in ("", "-wal", "-shm"):
        try:
            Path(f"{_TEST_DB_FILE}{suffix}").unlink(missing_ok=True)
        except OSError:
            # Windows keeps the handle until every pooled connection is gone;
            # the next run unlinks it up front anyway.
            pass


@pytest.fixture(scope="session", autouse=True)
def _create_schema() -> Generator[None, None, None]:
    _TEST_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    if _IS_SQLITE:
        _rm_sqlite_files()
    Base.metadata.create_all(_engine)
    yield
    Base.metadata.drop_all(_engine)
    _engine.dispose()
    # The app's own engine (app.db.session) is what eager tasks / real_db_session
    # use — dispose it too so the SQLite file handle is released.
    from app.db.session import engine as _app_engine

    _app_engine.dispose()
    shutil.rmtree(_TEST_STORAGE_DIR, ignore_errors=True)
    if _IS_SQLITE:
        _rm_sqlite_files()


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


@pytest.fixture(autouse=True)
def _restore_integration_credentials() -> Generator[None, None, None]:
    """Connecting Kaggle/Modal mutates `settings.*` and `os.environ` in place
    (the SDKs read env directly), and those mutations are process-global —
    the `client` fixture's DB rollback does not undo them. Without this,
    `test_modal_integration` connecting Modal leaves `settings.modal_configured`
    true for every later test (e.g. `test_training_providers...` then sees
    MODAL as available). Snapshot and restore around each test."""
    from app.core.config import settings as _s

    keys = ("KAGGLE_USERNAME", "KAGGLE_KEY", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")
    saved_settings = {k: getattr(_s, k) for k in keys}
    saved_env = {k: os.environ.get(k) for k in keys}
    try:
        yield
    finally:
        for k, v in saved_settings.items():
            setattr(_s, k, v)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


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


def _jpeg_bytes() -> bytes:
    from io import BytesIO

    from PIL import Image as PILImage

    img = PILImage.new("RGB", (64, 48), color=(80, 80, 80))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def version_and_base_model(real_client: TestClient, unique_name: str, tmp_path) -> tuple[str, str, str]:
    """A real project/dataset/approved-annotated-image/dataset-version plus
    a registered (fake-weights) base model — the minimum a training job
    needs to be created. Shared by test_training_jobs.py (LOCAL) and
    test_kaggle_training_poll.py (KAGGLE), which both just need a valid
    dataset_version_id/base_model_id and don't care how they got made."""
    project = real_client.post("/api/v1/projects", json={"name": unique_name}).json()
    real_client.patch(
        f"/api/v1/projects/{project['id']}",
        json={"class_config": [{"id": 0, "name": "ball"}, {"id": 1, "name": "cone"}, {"id": 2, "name": "cone_1"}]},
    )
    dataset = real_client.post(f"/api/v1/projects/{project['id']}/datasets", json={"name": "d"}).json()

    image = real_client.post(
        f"/api/v1/datasets/{dataset['id']}/images", files={"file": ("f.jpg", _jpeg_bytes(), "image/jpeg")}
    ).json()
    real_client.post(f"/api/v1/images/{image['id']}/approve")
    real_client.post(
        "/api/v1/annotations",
        json={"image_id": image["id"], "class_id": 1, "class_name": "cone", "x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.3},
    )
    version = real_client.post(f"/api/v1/datasets/{dataset['id']}/versions", json={}).json()

    weights = tmp_path / "detect_v1.pt"
    # Unique per fixture call — real_client commits for real across a
    # shared test-session database, and this fixture is reused by many
    # tests; a fixed name relied on `models` having no (name, version)
    # uniqueness (audit finding DB-03, now enforced).
    base_model = real_client.post(
        "/api/v1/models",
        json={"name": f"detect-{uuid.uuid4().hex[:8]}", "weights_path": str(weights), "kind": "DETECTOR"},
    ).json()

    return project["id"], version["id"], base_model["id"]
