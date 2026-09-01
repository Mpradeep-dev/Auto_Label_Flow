"""Periodic-job scheduler for the desktop app's `local` task runtime.

In `celery` mode the three periodics in `celery_app.PERIODIC_SCHEDULE` are
driven by Celery Beat (the worker's `-B` flag). In `local` mode there is no
Beat, so an APScheduler `BackgroundScheduler` started from the FastAPI
lifespan drives them instead. Each firing just calls the task's `.delay()`,
which lands it on the in-process `default` thread pool — the scheduler
thread itself never runs job work.

Kaggle/Modal pollers are only registered when their integration is
configured; unconfigured, they self-disable anyway and the wakeups are
pointless.
"""
from __future__ import annotations

import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

_scheduler = None  # apscheduler BackgroundScheduler | None


def start_scheduler() -> None:
    """Idempotent. No-op in `celery` mode (Beat handles it) and in tests."""
    global _scheduler
    if settings.ALF_TASK_QUEUE == "celery" or settings.ENV == "test":
        return
    if _scheduler is not None:
        return

    from apscheduler.schedulers.background import BackgroundScheduler

    # Importing the task modules registers them with the LocalCeleryShim.
    from app.workers.tasks.kaggle_training import poll_kaggle_training_jobs
    from app.workers.tasks.modal_training import poll_modal_training_jobs
    from app.workers.tasks.reconcile import reconcile_stale_jobs

    sched = BackgroundScheduler(timezone="UTC")
    sched.add_job(
        reconcile_stale_jobs.delay, "interval", seconds=300, id="reconcile-stale-jobs",
        max_instances=1, coalesce=True,
    )
    if settings.kaggle_configured:
        sched.add_job(
            poll_kaggle_training_jobs.delay, "interval", seconds=30,
            id="poll-kaggle-training-jobs", max_instances=1, coalesce=True,
        )
    if settings.modal_configured:
        sched.add_job(
            poll_modal_training_jobs.delay, "interval", seconds=120,
            id="poll-modal-training-jobs", max_instances=1, coalesce=True,
        )
    sched.start()
    _scheduler = sched
    logger.info("in-process job scheduler started (%d periodic jobs)", len(sched.get_jobs()))


def shutdown_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
