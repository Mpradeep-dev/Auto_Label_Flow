from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router
from app.core.config import settings


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # Replays previously-connected Kaggle and Modal accounts into the process
    # environment (see services/integrations/kaggle_connect.py and
    # modal_connect.py docstrings) so a container restart doesn't silently
    # lose connections until someone notices training stopped being offered.
    from app.db.session import SessionLocal
    from app.services.integrations import kaggle_connect, modal_connect

    db = SessionLocal()
    try:
        kaggle_connect.load_on_startup(db)
        modal_connect.load_on_startup(db)
    finally:
        db.close()
    yield


app = FastAPI(title=settings.APP_NAME, debug=settings.DEBUG, lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

# Local-storage dev convenience: serves whatever the local ObjectStorage
# backend wrote under LOCAL_STORAGE_DIR at /media/<key>, matching
# LocalFileStorage.get_url(). No-op in prod (STORAGE_BACKEND=minio serves
# via presigned URLs instead), but always mounted so dev/test never needs a
# config toggle to see uploaded media.
app.mount("/media", StaticFiles(directory=str(settings.LOCAL_STORAGE_DIR)), name="media")
