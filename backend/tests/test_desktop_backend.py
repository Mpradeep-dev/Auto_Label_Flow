"""Desktop-profile backend surface: SQLite dialect quirks, the Redis-free
health/system endpoints, the in-process progress store, and the BE-03
partial index surviving the `create_all` schema path."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
from sqlalchemy import inspect

from app.db.session import engine
from app.workers.progress import clear_cancel, is_cancel_requested, request_cancel
from app.workers.progress_store import InMemoryStore


def test_health_is_redis_free_and_ok_in_tests(client: TestClient) -> None:
    # The suite is offline regardless of ALF_TASK_QUEUE (see health.py).
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok"
    assert "redis" not in body
    assert body["version"]  # APP_VERSION always resolves to something
    assert "task_queue" in body


def test_system_info_shape(client: TestClient) -> None:
    body = client.get("/api/v1/system/info").json()
    for key in (
        "app_version",
        "schema_version",
        "task_queue",
        "storage_backend",
        "python_version",
        "gpu_pack_installed",
        "integrations_pack_installed",
    ):
        assert key in body
    # SQLite in tests → user_version is stamped by init_db / create_all path.
    assert body["gpu_pack_installed"] is False
    assert body["integrations_pack_installed"] is False


def test_system_packs_listing(client: TestClient) -> None:
    packs = client.get("/api/v1/system/packs").json()["packs"]
    assert {p["name"] for p in packs} == {"gpu", "integrations"}
    assert all(p["installed"] is False for p in packs)


def test_pack_install_rejected_without_data_dir(client: TestClient) -> None:
    # No ALF_DATA_DIR in the test env → packs are a desktop-only feature.
    resp = client.post("/api/v1/system/packs/gpu/install")
    assert resp.status_code == 400


def test_reconcile_stale_jobs_runs_on_sqlite() -> None:
    """Regression: `updated_at < datetime.now(timezone.utc)` raised
    `TypeError: can't compare offset-naive and offset-aware datetimes` on
    SQLite before TZDateTime. Just running the sweep must not raise."""
    from app.workers.tasks.reconcile import reconcile_stale_jobs

    counts = reconcile_stale_jobs()
    assert set(counts) == {"inference", "training", "video", "roboflow"}


def test_tzdatetime_roundtrips_aware_on_sqlite(db_session) -> None:
    """A tz-aware datetime written through the ORM comes back tz-aware and
    comparable to `datetime.now(timezone.utc)` — the property reconcile relies
    on."""
    from app.models.project import Project

    p = Project(name="tz", slug=f"tz-{datetime.now().timestamp()}", class_config=[])
    db_session.add(p)
    db_session.flush()
    db_session.refresh(p)
    assert p.created_at.tzinfo is not None
    # must not raise
    _ = p.created_at < datetime.now(timezone.utc) + timedelta(days=1)


def test_be03_partial_unique_index_exists_on_sqlite() -> None:
    idx = {i["name"] for i in inspect(engine).get_indexes("inference_jobs")}
    assert "uq_inference_jobs_one_active_per_dataset" in idx


def test_in_memory_progress_store_roundtrip() -> None:
    store = InMemoryStore()
    store.set("k", "v", 60)
    assert store.get("k") == "v"
    assert store.exists("k")
    store.delete("k")
    assert store.get("k") is None
    assert not store.exists("k")


def test_cancel_flag_via_store() -> None:
    request_cancel("job-xyz")
    assert is_cancel_requested("job-xyz")
    clear_cancel("job-xyz")
    assert not is_cancel_requested("job-xyz")
