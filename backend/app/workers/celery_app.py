"""Job execution config + the `celery_app` object every task module decorates.

Two queues (PLAN "Jobs: Celery + Redis, two queues"):

  - `gpu`: anything touching CUDA (inference, local training). Run at
    concurrency=1 — with 8GB VRAM, a training run and an inference batch
    concurrently will OOM. Serializing by construction beats app-level
    locking.
  - `default`: frame extraction, export, housekeeping — normal concurrency.

**Two runtimes, one decorator surface.** `settings.ALF_TASK_QUEUE` picks:

  - `celery` (docker-compose / server): a real `celery.Celery` with a Redis
    broker. `docker-compose.yml`'s `worker` service runs both queues in one
    process (`-Q gpu,default -c 1 -B`); a production deploy would split them.
  - `local` (default; desktop app): `LocalCeleryShim` — two in-process
    `ThreadPoolExecutor`s, no broker, no `celery` import. The periodic
    `beat_schedule` below is driven by `workers/scheduler.py` (APScheduler)
    instead of Celery Beat.

Task modules and `.delay()` call sites are identical either way.
"""
from __future__ import annotations

from app.core.config import settings

_USE_CELERY = settings.ALF_TASK_QUEUE == "celery"

# Per-task-type time limits (audit finding REL-01). Before these existed,
# nothing bounded how long a task could run — a hung/stuck task (network
# call that never returns, a model load that spins forever) held its DB row
# at RUNNING permanently, with no timeout and no worker-side signal that
# anything was wrong. Soft limit raises `SoftTimeLimitExceeded` *inside* the
# task, which each task's existing `except Exception` already catches and
# turns into a clean FAILED status; the hard limit (a short grace period
# later) forcibly kills the task if it didn't exit on its own, as a last
# resort — that path bypasses Python exception handling entirely, which is
# exactly why RECONCILE_STALE_AFTER_S below exists as a second, independent
# safety net that doesn't depend on the task getting a chance to clean up
# after itself at all (e.g. the whole worker process being killed).
#
# Values are generous ceilings meant to catch "this is clearly never
# finishing," not normal completion times — training in particular can
# legitimately run for a long time depending on epochs/dataset size, so its
# limit is deliberately loose.
INFERENCE_TIME_LIMIT_S = 30 * 60
INFERENCE_SOFT_TIME_LIMIT_S = 25 * 60
TRAINING_TIME_LIMIT_S = 6 * 60 * 60
TRAINING_SOFT_TIME_LIMIT_S = 6 * 60 * 60 - 120
QUALITY_TIME_LIMIT_S = 25 * 60
QUALITY_SOFT_TIME_LIMIT_S = 20 * 60
VIDEO_TIME_LIMIT_S = 20 * 60
VIDEO_SOFT_TIME_LIMIT_S = 15 * 60
ROBOFLOW_TIME_LIMIT_S = 35 * 60
ROBOFLOW_SOFT_TIME_LIMIT_S = 30 * 60
# Azure-Blob import walks a whole container prefix (production sets run to
# tens of thousands of images) doing a small header read + label read per
# image — no full-image downloads, but still long. Generous ceiling meant
# only to catch "this is clearly hung," not normal completion.
BLOB_IMPORT_TIME_LIMIT_S = 3 * 60 * 60
BLOB_IMPORT_SOFT_TIME_LIMIT_S = 3 * 60 * 60 - 120
# The dataset-upload readiness poll alone (kaggle_provider.py's
# `_push_kernel`) can take up to 5 minutes; this bounds the whole
# export/zip/upload/push sequence generously above that without leaving it
# effectively unbounded on the `default` queue if a Kaggle API call hangs.
KAGGLE_START_TIME_LIMIT_S = 10 * 60
KAGGLE_START_SOFT_TIME_LIMIT_S = 10 * 60 - 30
# Modal: dataset zip + Volume upload + function submission — typically
# faster than Kaggle (no readiness poll), but generous to handle large
# datasets and cold Volume creation.
MODAL_START_TIME_LIMIT_S = 15 * 60
MODAL_START_SOFT_TIME_LIMIT_S = 15 * 60 - 30

# How long each job type can sit in QUEUED/RUNNING with no `updated_at`
# progress before the reconciliation sweep (tasks/reconcile.py) considers it
# dead rather than just slow, and fails it — each task's own hard time
# limit above, plus a buffer wider than the sweep's own 5-minute cadence so
# a job isn't flagged the instant its normal time limit passes.
RECONCILE_STALE_AFTER_S = {
    "inference": INFERENCE_TIME_LIMIT_S + 10 * 60,
    "training": TRAINING_TIME_LIMIT_S + 10 * 60,
    "video": VIDEO_TIME_LIMIT_S + 10 * 60,
    "roboflow": ROBOFLOW_TIME_LIMIT_S + 10 * 60,
    "blob_import": BLOB_IMPORT_TIME_LIMIT_S + 10 * 60,
}

# The three periodic jobs. In `celery` mode this feeds Celery Beat's
# `beat_schedule`; in `local` mode `workers/scheduler.py` reads it directly.
PERIODIC_SCHEDULE = {
    "reconcile-stale-jobs": {
        "task": "app.workers.tasks.reconcile.reconcile_stale_jobs",
        "schedule": 300.0,
    },
    "poll-kaggle-training-jobs": {
        "task": "app.workers.tasks.kaggle_training.poll_kaggle_training_jobs",
        "schedule": 30.0,
    },
    "poll-modal-training-jobs": {
        "task": "app.workers.tasks.modal_training.poll_modal_training_jobs",
        "schedule": 120.0,
    },
}

if not _USE_CELERY:
    from app.workers.local_queue import LocalCeleryShim

    celery_app = LocalCeleryShim()
else:
    from celery import Celery
    from celery.signals import worker_process_init

    celery_app = Celery(
        "annotate",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
        include=[
            "app.workers.tasks.inference",
            "app.workers.tasks.video",
            "app.workers.tasks.training",
            "app.workers.tasks.kaggle_training",
            "app.workers.tasks.modal_training",
            "app.workers.tasks.quality",
            "app.workers.tasks.roboflow",
            "app.workers.tasks.blob_import",
            "app.workers.tasks.reconcile",
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
            "app.workers.tasks.roboflow.*": {"queue": "default"},
            "app.workers.tasks.blob_import.*": {"queue": "default"},
            "app.workers.tasks.reconcile.*": {"queue": "default"},
            # Polling is just Kaggle API calls, no local CUDA work — belongs
            # on `default`, not `gpu` (concurrency=1, shouldn't wait on
            # network round trips to Kaggle).
            "app.workers.tasks.kaggle_training.*": {"queue": "default"},
            # Modal: Volume uploads + status checks are CPU/network, not GPU.
            "app.workers.tasks.modal_training.*": {"queue": "default"},
        },
        task_track_started=True,
        worker_prefetch_multiplier=1,  # long-running tasks: don't hoard extra work per worker
        # Second half of REL-01's fix: sweeps for jobs stuck in QUEUED/RUNNING
        # with no recent progress (see RECONCILE_STALE_AFTER_S) and fails them
        # cleanly — independent of whether the stuck task ever gets a chance
        # to clean up after itself.
        beat_schedule=PERIODIC_SCHEDULE,
    )

    if settings.ENV == "test":
        # Run tasks synchronously in-process, no broker/worker required —
        # keeps the test suite offline while exercising the real task code.
        celery_app.conf.update(task_always_eager=True, task_eager_propagates=True)

    @worker_process_init.connect
    def _load_kaggle_credentials(**_kwargs) -> None:
        """The FastAPI process replays a connected Kaggle account into its
        own environment at startup (see `main.py`'s lifespan). The Celery
        worker is a separate process with its own empty environment, so a
        task needing Kaggle credentials found none there even when Kaggle was
        connected. `worker_process_init` fires once per pool worker process,
        mirroring the FastAPI lifespan hook for this process instead."""
        from app.db.session import SessionLocal
        from app.services.integrations import kaggle_connect

        db = SessionLocal()
        try:
            kaggle_connect.load_on_startup(db)
        finally:
            db.close()

    @worker_process_init.connect
    def _load_modal_credentials(**_kwargs) -> None:
        """Same reasoning as _load_kaggle_credentials, for Modal credentials
        used by modal_training.py's poll task."""
        from app.db.session import SessionLocal
        from app.services.integrations import modal_connect

        db = SessionLocal()
        try:
            modal_connect.load_on_startup(db)
        finally:
            db.close()
