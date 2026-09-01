"""In-process task queue — the desktop app's replacement for Celery + Redis.

`LocalCeleryShim` is a drop-in stand-in for the `celery_app` object: task
modules keep decorating with `@celery_app.task(bind=True, name=..., time_limit=...)`
and call sites keep calling `.delay(...)`, but nothing imports `celery` and
nothing talks to a broker.

Execution model, matching the two Celery queues:
  - `gpu`     queue → `ThreadPoolExecutor(max_workers=1)` — inference /
    training / quality run one at a time (the VRAM-OOM-avoidance invariant;
    on CPU it just stops the box thrashing).
  - `default` queue → `ThreadPoolExecutor(max_workers=3)` — video frame
    extraction, roboflow, reconcile, remote-training polls.

Threads, not processes: torch / OpenCV release the GIL during compute, and a
Windows `ProcessPoolExecutor` re-runs every import side effect (config,
torch init) per task.

In `ENV=test` every `.delay()` runs inline and synchronously, exactly like
Celery's `task_always_eager` — so the existing eager-task tests are
unchanged.
"""
from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

from app.core.config import settings

logger = logging.getLogger(__name__)

# Task-name substring → queue. Mirrors `celery_app.py`'s `task_routes`.
_GPU_QUEUE_MARKERS = (".inference.", ".training.", ".quality.")


def _queue_for(name: str) -> str:
    return "gpu" if any(m in name for m in _GPU_QUEUE_MARKERS) else "default"


class _Request:
    """The subset of a Celery task's `self.request` that task bodies use."""

    def __init__(self, task_id: str) -> None:
        self.id = task_id


class LocalTask:
    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        shim: "LocalCeleryShim",
        name: str,
        bind: bool = False,
        soft_time_limit: int | None = None,
        time_limit: int | None = None,
    ) -> None:
        self.fn = fn
        self.name = name
        self.bind = bind
        self.soft_time_limit = soft_time_limit
        self.time_limit = time_limit
        self._shim = shim
        self._local = threading.local()
        self.__name__ = getattr(fn, "__name__", name)
        self.__doc__ = fn.__doc__

    # --- direct call (eager path / scheduler calling the body directly) ---
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._invoke(args, kwargs, task_id=uuid.uuid4().hex)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        return self.__call__(*args, **kwargs)

    @property
    def request(self) -> _Request:
        req = getattr(self._local, "request", None)
        if req is None:
            req = _Request(uuid.uuid4().hex)
            self._local.request = req
        return req

    def _invoke(self, args: tuple, kwargs: dict, *, task_id: str) -> Any:
        self._local.request = _Request(task_id)
        watchdog = self._start_watchdog(args)
        try:
            if self.bind:
                return self.fn(self, *args, **kwargs)
            return self.fn(*args, **kwargs)
        finally:
            if watchdog is not None:
                watchdog.cancel()
            self._local.request = None

    def _start_watchdog(self, args: tuple) -> threading.Timer | None:
        if not self.soft_time_limit:
            return None
        first = args[0] if args else None

        def _on_soft_timeout() -> None:
            logger.warning("task %s exceeded soft time limit (%ss)", self.name, self.soft_time_limit)
            if isinstance(first, str):
                try:  # best-effort cooperative cancel; reconcile is the real backstop
                    from app.workers.progress import request_cancel

                    request_cancel(first)
                except Exception:  # noqa: BLE001
                    pass

        t = threading.Timer(self.soft_time_limit, _on_soft_timeout)
        t.daemon = True
        t.start()
        return t

    # --- async dispatch ---
    def delay(self, *args: Any, **kwargs: Any) -> "LocalAsyncResult":
        return self.apply_async(args=args, kwargs=kwargs)

    def apply_async(
        self,
        args: tuple | list | None = None,
        kwargs: dict | None = None,
        **_opts: Any,
    ) -> "LocalAsyncResult":
        args = tuple(args or ())
        kwargs = dict(kwargs or {})
        task_id = uuid.uuid4().hex

        if settings.ENV == "test":
            self._invoke(args, kwargs, task_id=task_id)
            return LocalAsyncResult(task_id, done=True)

        executor = self._shim.executor(_queue_for(self.name))

        def _job() -> Any:
            try:
                return self._invoke(args, kwargs, task_id=task_id)
            except Exception:  # noqa: BLE001 — task bodies already persist FAILED; log for the app console
                logger.exception("background task %s failed", self.name)

        future = executor.submit(_job)
        return LocalAsyncResult(task_id, future=future)


class LocalAsyncResult:
    def __init__(self, task_id: str, *, future: Future | None = None, done: bool = False) -> None:
        self.id = task_id
        self.task_id = task_id
        self._future = future
        self._done = done

    def ready(self) -> bool:
        if self._done:
            return True
        return self._future.done() if self._future is not None else False

    def get(self, timeout: float | None = None) -> Any:
        if self._future is not None:
            return self._future.result(timeout=timeout)
        return None


class LocalCeleryShim:
    """Stands in for `celery.Celery` in `local` mode."""

    def __init__(self) -> None:
        self.tasks: dict[str, LocalTask] = {}
        self._executors: dict[str, ThreadPoolExecutor] = {}
        self._lock = threading.Lock()
        self.conf = _Conf()

    def executor(self, queue: str) -> ThreadPoolExecutor:
        with self._lock:
            ex = self._executors.get(queue)
            if ex is None:
                workers = 1 if queue == "gpu" else 3
                ex = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"alf-{queue}")
                self._executors[queue] = ex
            return ex

    def task(self, *dargs: Any, **dkwargs: Any):
        # Support both @celery_app.task and @celery_app.task(name=..., bind=True, ...)
        def _wrap(fn: Callable[..., Any]) -> LocalTask:
            name = dkwargs.get("name") or f"{fn.__module__}.{fn.__name__}"
            task = LocalTask(
                fn,
                shim=self,
                name=name,
                bind=bool(dkwargs.get("bind", False)),
                soft_time_limit=dkwargs.get("soft_time_limit"),
                time_limit=dkwargs.get("time_limit"),
            )
            self.tasks[name] = task
            return task

        if dargs and callable(dargs[0]) and not dkwargs:
            return _wrap(dargs[0])
        return _wrap

    def send_task(self, name: str, args: tuple | list | None = None, **_opts: Any) -> LocalAsyncResult:
        task = self.tasks.get(name)
        if task is None:
            raise KeyError(f"unknown task {name!r}")
        return task.apply_async(args=tuple(args or ()))

    def shutdown(self) -> None:
        with self._lock:
            for ex in self._executors.values():
                ex.shutdown(wait=False, cancel_futures=True)
            self._executors.clear()


class _Conf:
    """No-op stand-in for `celery_app.conf` — `.update(**kw)` is a sink."""

    def update(self, *_a: Any, **_kw: Any) -> None:
        return None
