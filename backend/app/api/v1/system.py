"""Desktop-app system info + optional add-on packs.

`/system/info` backs the Settings "About" panel and support diagnostics.
`/system/packs` + `/system/packs/{name}/install` back the "Download GPU
training support" / "Download cloud integrations" buttons. On the server
deployment these endpoints still respond (packs simply report not-installed
and install is rejected) — they are only actionable in the packaged app.
"""
from __future__ import annotations

import asyncio
import json
import platform
import sys

from fastapi import APIRouter, HTTPException, Path
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.db.session import engine
from app.schemas.sam import SamModelStatusRead
from app.services.system import packs, sam_models

router = APIRouter(prefix="/system", tags=["system"])


def _schema_version() -> int | None:
    if engine.dialect.name != "sqlite":
        return None
    with engine.connect() as conn:
        return conn.exec_driver_sql("PRAGMA user_version").scalar_one()


@router.get("/info")
def system_info() -> dict:
    return {
        "app_version": settings.APP_VERSION,
        "schema_version": _schema_version(),
        "task_queue": settings.ALF_TASK_QUEUE,
        "storage_backend": settings.STORAGE_BACKEND,
        "data_dir": settings.ALF_DATA_DIR,
        "python_version": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "frozen": bool(getattr(sys, "frozen", False)),
        "gpu_pack_installed": packs.is_installed("gpu"),
        "integrations_pack_installed": packs.is_installed("integrations"),
    }


@router.get("/packs")
def list_packs() -> dict:
    return {
        "packs": [
            {
                "name": s.name,
                "installed": s.installed,
                "version": s.version,
                "size_bytes": s.size_bytes,
            }
            for s in packs.all_status()
        ]
    }


@router.post("/packs/{name}/install", status_code=202)
def install_pack(name: str = Path(pattern="^(gpu|integrations)$")) -> dict:
    if not settings.ALF_DATA_DIR:
        raise HTTPException(
            status_code=400,
            detail="Add-on packs are only available in the desktop app.",
        )
    from app.workers.tasks.packs import install_pack_task

    result = install_pack_task.delay(name)
    return {"pack": name, "task_id": result.id}


@router.get("/packs/{name}/stream")
async def stream_pack_install(name: str = Path(pattern="^(gpu|integrations)$")):
    """SSE stream of the running pack install (text lines from pip)."""
    from app.workers.tasks.packs import get_install_progress

    async def event_stream():
        last = None
        for _ in range(3600):
            prog = get_install_progress(name) or {"state": "idle", "detail": "", "lines": []}
            payload = json.dumps(prog)
            if payload != last:
                yield f"data: {payload}\n\n"
                last = payload
            if prog["state"] in ("done", "failed", "idle"):
                if prog["state"] != "idle":
                    return
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.delete("/packs/{name}", status_code=204)
def remove_pack(name: str = Path(pattern="^(gpu|integrations)$")) -> None:
    d = packs.pack_dir(name)
    if d is not None and d.is_dir():
        import shutil

        shutil.rmtree(d, ignore_errors=True)


# --- Optional SAM checkpoints ("Download SAM Lite/Full" on Settings) ---
#
# Deliberately a separate mechanism from /packs above: a pack installs a
# pip dependency set into sys.path, but ultralytics (already a base
# dependency) ships SAM/MobileSAM support natively — what's optional here
# is only the checkpoint *file*, tracked by services/system/sam_models.py
# (present/absent on disk under MODELS_DIR/sam/, not a pack.json marker).


@router.get("/sam-models", response_model=list[SamModelStatusRead])
def list_sam_models() -> list[sam_models.SamModelStatus]:
    return sam_models.all_status()


@router.post("/sam-models/{name}/download", status_code=202)
def install_sam_model(name: str = Path(pattern="^(sam-lite|sam-full)$")) -> dict:
    from app.workers.tasks.sam_download import download_sam_model

    result = download_sam_model.delay(name)
    return {"variant": name, "task_id": result.id}


@router.get("/sam-models/{name}/stream")
async def stream_sam_model_download(name: str = Path(pattern="^(sam-lite|sam-full)$")):
    """SSE stream of download progress — byte counts, unlike /packs/{name}/stream's
    pip-log lines, since this is a plain file download."""
    from app.workers.tasks.sam_download import get_download_progress

    async def event_stream():
        last = None
        for _ in range(3600):
            prog = get_download_progress(name) or {"state": "idle", "detail": "", "downloaded": 0, "total": 0}
            payload = json.dumps(prog)
            if payload != last:
                yield f"data: {payload}\n\n"
                last = payload
            if prog["state"] in ("done", "failed"):
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.delete("/sam-models/{name}", status_code=204)
def remove_sam_model(name: str = Path(pattern="^(sam-lite|sam-full)$")) -> None:
    sam_models.remove(name)
