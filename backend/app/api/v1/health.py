from __future__ import annotations

import redis
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
    # silently never run — the one thing most likely to quietly break a
    # session went completely unsignaled.
    db_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    redis_ok = True
    try:
        redis.from_url(settings.REDIS_URL, socket_connect_timeout=2, socket_timeout=2).ping()
    except Exception:
        redis_ok = False

    storage_ok = True
    try:
        get_storage().exists("__healthcheck__")
    except Exception:
        storage_ok = False

    all_ok = db_ok and redis_ok and storage_ok
    return {
        "status": "ok" if all_ok else "degraded",
        "app": settings.APP_NAME,
        "env": settings.ENV,
        "database": "ok" if db_ok else "unreachable",
        "redis": "ok" if redis_ok else "unreachable",
        "storage": "ok" if storage_ok else "unreachable",
        "storage_backend": settings.STORAGE_BACKEND,
        "kaggle_configured": settings.kaggle_configured,
    }
