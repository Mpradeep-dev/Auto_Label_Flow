from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.services.storage.factory import get_storage

router = APIRouter(tags=["health"])


@router.get("/health")
def health(db: Session = Depends(get_db)) -> dict:
    # Audit finding BE-05: this used to check Postgres only. A dead Redis
    # broker (or an unreachable storage backend) still reported "ok" here
    # while every job-creation endpoint kept accepting requests that would
    # silently never run.
    db_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    # Redis is only in the picture under the `celery` task runtime. The
    # desktop app (`local`) has no Redis; reporting it "unreachable" there
    # would wedge the Electron health-poll on "degraded" forever. The test
    # suite stays offline regardless of task_queue.
    uses_redis = settings.ALF_TASK_QUEUE == "celery" and settings.ENV != "test"
    redis_ok = True
    if uses_redis:
        try:
            import redis

            redis.from_url(settings.REDIS_URL, socket_connect_timeout=2, socket_timeout=2).ping()
        except Exception:
            redis_ok = False

    storage_ok = True
    try:
        get_storage().exists("__healthcheck__")
    except Exception:
        storage_ok = False

    all_ok = db_ok and redis_ok and storage_ok
    result = {
        "status": "ok" if all_ok else "degraded",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "env": settings.ENV,
        "task_queue": settings.ALF_TASK_QUEUE,
        "database": "ok" if db_ok else "unreachable",
        "storage": "ok" if storage_ok else "unreachable",
        "storage_backend": settings.STORAGE_BACKEND,
        "kaggle_configured": settings.kaggle_configured,
    }
    if uses_redis:
        result["redis"] = "ok" if redis_ok else "unreachable"
    return result
