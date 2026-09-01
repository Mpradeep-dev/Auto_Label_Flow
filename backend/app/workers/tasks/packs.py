"""Add-on pack installer — downloads and installs the optional GPU-training
and cloud-integrations dependency sets into a per-pack `site-packages` under
the user data dir. Runs on the `default` queue.

Implementation is a `pip install --target` against a pinned spec list. pip's
progress is streamed as text lines into the progress store under
`pack:install:<name>`, which `/api/v1/system/packs/{name}/stream` relays over
SSE.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone

from app.services.system import packs
from app.workers.celery_app import celery_app
from app.workers.progress_store import get_store

# Pinned install specs per pack. Kept here (not requirements files) because
# the GPU set needs a custom index and both are resolved at install time on
# the user's machine, not at build time.
_PACK_SPECS: dict[str, list[str]] = {
    "gpu": [
        "--index-url", "https://download.pytorch.org/whl/cu121",
        "torch==2.5.1", "torchvision==0.20.1",
    ],
    "integrations": [
        "kaggle==1.6.17",
        "modal>=0.64.0",
        "roboflow==1.4.1",
        "boto3==1.35.71",
        "azure-storage-blob==12.24.0",
    ],
}

_PROGRESS_KEY = "pack:install:{name}"
_TTL_S = 3600


def _emit(name: str, state: str, detail: str, lines: list[str]) -> None:
    get_store().set(
        _PROGRESS_KEY.format(name=name),
        json.dumps({"state": state, "detail": detail, "lines": lines[-40:]}),
        _TTL_S,
    )


def get_install_progress(name: str) -> dict | None:
    raw = get_store().get(_PROGRESS_KEY.format(name=name))
    return json.loads(raw) if raw else None


@celery_app.task(name="app.workers.tasks.packs.install_pack_task")
def install_pack_task(name: str) -> None:
    spec = _PACK_SPECS.get(name)
    target = packs._site_packages(name)  # noqa: SLF001 — internal layout helper
    if spec is None or target is None:
        _emit(name, "failed", f"unknown pack or no data dir: {name}", [])
        return

    target.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    _emit(name, "running", "starting installer", lines)

    cmd = [sys.executable, "-m", "pip", "install", "--no-input", "--target", str(target), *spec]
    proc = subprocess.Popen(  # noqa: S603
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    assert proc.stdout is not None
    last_emit = 0.0
    for line in proc.stdout:
        lines.append(line.rstrip())
        now = time.monotonic()
        if now - last_emit > 0.5:
            _emit(name, "running", lines[-1][:200], lines)
            last_emit = now
    code = proc.wait()

    if code != 0:
        _emit(name, "failed", f"pip exited {code}", lines)
        return

    (target.parent / "pack.json").write_text(
        json.dumps(
            {
                "name": name,
                "version": "1",
                "specs": spec,
                "installed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    )
    packs.activate_installed_packs()
    _emit(name, "done", "installed", lines)
