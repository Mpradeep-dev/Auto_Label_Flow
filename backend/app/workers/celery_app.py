"""Celery app with two queues (PLAN "Jobs: Celery + Redis, two queues"):

  - `gpu`: anything touching CUDA (inference, local training). Run this
    queue's worker at concurrency=1 — with 8GB VRAM, a training run and an
    inference batch running concurrently will OOM. Serializing by
    construction beats app-level locking.
  - `default`: frame extraction, export, housekeeping — normal concurrency.

`docker-compose.yml`'s `worker` service runs both queues in one process for
local dev (`-Q gpu,default -c 1`); a production deploy would split them into
separate services/replicas so `default` work isn't held hostage by GPU
concurrency=1.
"""
from __future__ import annotations

from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "annotate",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.workers.tasks.inference",
        "app.workers.tasks.video",
        "app.workers.tasks.training",
        "app.workers.tasks.quality",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_routes={
        "app.workers.tasks.inference.*": {"queue": "gpu"},
        "app.workers.tasks.training.*": {"queue": "gpu"},
        "app.workers.tasks.quality.*": {"queue": "gpu"},
        "app.workers.tasks.video.*": {"queue": "default"},
    },
    task_track_started=True,
    worker_prefetch_multiplier=1,  # long-running tasks: don't hoard extra work per worker
)

if settings.ENV == "test":
    # Run tasks synchronously in-process, no broker/worker required — keeps
    # the test suite offline (house convention) while still exercising the
    # real task code, not a mock of it.
    celery_app.conf.update(task_always_eager=True, task_eager_propagates=True)
