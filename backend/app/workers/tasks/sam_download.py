"""Downloads a SAM/MobileSAM checkpoint into `MODELS_DIR/sam/`, streaming
byte progress into the progress store — same store + SSE-relay pattern
`workers/tasks/packs.py` uses for its pip-install log, but progress here is
a real downloaded/total byte count since this is a plain file download, not
a subprocess log. Runs on the `default` queue (network-bound, not GPU).

Redirect-following mirrors `services/inference/registry.py::_download_weights`
exactly: every hop re-checked by `core.security.assert_public_host` (SEC-02),
not just the initial URL.
"""
from __future__ import annotations

import json
import time
from urllib.parse import urljoin, urlsplit

import httpx

from app.core.security import UnsafeUrlError, assert_public_host
from app.services.system import sam_models
from app.workers.celery_app import celery_app
from app.workers.progress_store import get_store

_PROGRESS_KEY = "sam:download:{name}"
_TTL_S = 3600
_MAX_REDIRECTS = 5
_EMIT_THROTTLE_S = 0.5


class SamDownloadError(RuntimeError):
    pass


def _emit(name: str, state: str, detail: str, downloaded: int, total: int) -> None:
    get_store().set(
        _PROGRESS_KEY.format(name=name),
        json.dumps({"state": state, "detail": detail, "downloaded": downloaded, "total": total}),
        _TTL_S,
    )


def get_download_progress(name: str) -> dict | None:
    raw = get_store().get(_PROGRESS_KEY.format(name=name))
    return json.loads(raw) if raw else None


@celery_app.task(name="app.workers.tasks.sam_download.download_sam_model")
def download_sam_model(variant_name: str) -> None:
    variant = sam_models.SAM_VARIANTS.get(variant_name)
    if variant is None:
        _emit(variant_name, "failed", f"unknown SAM variant: {variant_name}", 0, 0)
        return

    dest = sam_models.weights_path(variant_name)
    tmp_dest = dest.parent / f"{dest.name}.part"
    _emit(variant_name, "running", "starting download", 0, 0)

    downloaded = 0
    total = 0
    try:
        current_url = variant.url
        scheme = urlsplit(current_url).scheme
        if scheme not in ("http", "https"):
            raise SamDownloadError(f"Unsupported URL scheme {scheme!r}")
        assert_public_host(current_url)

        with httpx.Client(follow_redirects=False, timeout=300.0) as client:
            for _ in range(_MAX_REDIRECTS + 1):
                with client.stream("GET", current_url) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            response.raise_for_status()
                            break
                        current_url = urljoin(current_url, location)
                        if urlsplit(current_url).scheme not in ("http", "https"):
                            raise SamDownloadError(f"Redirect to unsupported scheme: {current_url}")
                        assert_public_host(current_url)
                        continue

                    response.raise_for_status()
                    total = int(response.headers.get("content-length") or 0)
                    last_emit = 0.0
                    with tmp_dest.open("wb") as f:
                        for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                            f.write(chunk)
                            downloaded += len(chunk)
                            now = time.monotonic()
                            if now - last_emit > _EMIT_THROTTLE_S:
                                _emit(variant_name, "running", f"{downloaded} / {total or '?'} bytes", downloaded, total)
                                last_emit = now
                    break
            else:
                raise SamDownloadError(f"Too many redirects fetching {variant.url}")

        tmp_dest.replace(dest)
    except (httpx.HTTPError, UnsafeUrlError, SamDownloadError) as exc:
        tmp_dest.unlink(missing_ok=True)
        _emit(variant_name, "failed", str(exc), downloaded, total)
        return

    _emit(variant_name, "done", "installed", downloaded, total)
