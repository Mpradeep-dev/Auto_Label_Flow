from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router
from app.core.config import settings


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # Desktop app: put any installed optional packs (GPU torch, cloud SDKs)
    # on sys.path before torch / the SDKs might be imported.
    from app.services.system import packs

    packs.activate_installed_packs()

    # Desktop app: materialise the bundled SQLite schema on first run (no-op
    # on Postgres and once stamped). Must happen before anything reads the DB.
    from app.db.init_db import init_sqlite_schema
    from app.db.session import engine

    init_sqlite_schema(engine)

    # Replays previously-connected Kaggle and Modal accounts into the process
    # environment (see services/integrations/kaggle_connect.py and
    # modal_connect.py docstrings) so a restart doesn't silently lose
    # connections until someone notices training stopped being offered.
    from app.db.session import SessionLocal
    from app.services.integrations import kaggle_connect, modal_connect

    db = SessionLocal()
    try:
        kaggle_connect.load_on_startup(db)
        modal_connect.load_on_startup(db)
    finally:
        db.close()

    # `local` task runtime: start the in-process periodic scheduler (no-op
    # under Celery / in tests).
    from app.workers.scheduler import shutdown_scheduler, start_scheduler

    start_scheduler()
    try:
        yield
    finally:
        shutdown_scheduler()
        from app.workers.celery_app import celery_app

        if hasattr(celery_app, "shutdown"):
            celery_app.shutdown()


app = FastAPI(title=settings.APP_NAME, debug=settings.DEBUG, lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

# Local-storage media: serves whatever the local ObjectStorage backend wrote
# under LOCAL_STORAGE_DIR at /media/<key>, matching LocalFileStorage.get_url().
# No-op in prod with STORAGE_BACKEND=minio (presigned URLs instead), but
# always mounted so dev/test/desktop never needs a config toggle.
app.mount("/media", StaticFiles(directory=str(settings.LOCAL_STORAGE_DIR)), name="media")


# --- Desktop app: FastAPI also serves the built SPA, so the whole app is one
# origin (http://127.0.0.1:<port>). Dev/server leave FRONTEND_DIST_DIR unset
# and the Vite dev server serves the frontend. ---
if settings.FRONTEND_DIST_DIR is not None:
    _dist = settings.FRONTEND_DIST_DIR
    _assets = _dist / "assets"
    if _assets.is_dir():

        class _ImmutableStatic(StaticFiles):
            def is_not_modified(self, response_headers, request_headers) -> bool:  # noqa: ANN001
                return super().is_not_modified(response_headers, request_headers)

            def file_response(self, *args, **kwargs):  # noqa: ANN002, ANN003
                resp = super().file_response(*args, **kwargs)
                resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
                return resp

        app.mount("/assets", _ImmutableStatic(directory=str(_assets)), name="assets")

    _index = _dist / "index.html"

    @app.get("/{full_path:path}", include_in_schema=False)
    def _spa(full_path: str) -> FileResponse:
        # API / media / assets paths are handled above; a miss there is a real
        # 404, never the HTML shell (which would break the frontend fetch layer).
        if full_path.startswith(("api/", "media/", "assets/")):
            raise HTTPException(status_code=404)
        candidate = (_dist / full_path).resolve()
        if full_path and candidate.is_file() and _dist.resolve() in candidate.parents:
            return FileResponse(candidate)
        # Deep links / hard refreshes fall back to the shell; no-cache so a
        # new build's index.html is picked up after an app update.
        return FileResponse(_index, headers={"Cache-Control": "no-cache"})
