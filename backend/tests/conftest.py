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
