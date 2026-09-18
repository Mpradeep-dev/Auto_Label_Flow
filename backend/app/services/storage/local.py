"""Local-filesystem ObjectStorage — the dev-mode backend. Files under
`STORAGE_ROOT/<key>` are served by FastAPI's StaticFiles mount at
`/media/<key>` (see app/main.py)."""
from __future__ import annotations

import shutil
import threading
from pathlib import Path

from app.core.config import settings
from app.services.storage.base import ObjectStorage


class PathTraversalError(ValueError):
    """A storage key attempted to escape the storage root."""


class LocalFileStorage(ObjectStorage):
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or settings.LOCAL_STORAGE_DIR).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        # Guards `_resolve()` + the parent `mkdir(parents=True)` a write does
        # right after it — NOT the actual file write, which stays concurrent.
        # `Path.resolve()` on a path whose parent directories don't exist yet
        # walks the filesystem to find the longest existing prefix; on
        # Windows, racing that walk against another thread's concurrent
        # `mkdir(parents=True)` for a sibling key under the same not-yet-
        # created directory (e.g. several images of one dataset import,
        # `.../images/<id>.jpg`, landing on this backend at once) can
        # observe a transient, inconsistent filesystem state and resolve to
        # a path that spuriously fails the containment check below —
        # reproduced directly once callers started uploading concurrently
        # (Roboflow/COCO/CVAT import, each via a bounded thread pool).
        self._mkdir_lock = threading.Lock()

    def _resolve(self, key: str) -> Path:
        # Reject the key outright if it tries to escape via absolute paths
        # or ".." segments BEFORE touching the filesystem, then re-verify
        # with a resolved-path containment check as defense in depth.
        if key.startswith("/") or key.startswith("\\") or ".." in Path(key).parts:
            raise PathTraversalError(f"Unsafe storage key: {key!r}")
        candidate = (self.root / key).resolve()
        if self.root not in candidate.parents and candidate != self.root:
            raise PathTraversalError(f"Storage key escapes root: {key!r}")
        return candidate

    def _prepare_dest(self, key: str) -> Path:
        """`_resolve(key)` plus ensuring its parent directory exists, both
        under `_mkdir_lock` — see the lock's docstring in `__init__` for why
        this pair specifically needs serializing under concurrent writers."""
        with self._mkdir_lock:
            dest = self._resolve(key)
            dest.parent.mkdir(parents=True, exist_ok=True)
        return dest

    def upload(self, local_path: Path, key: str, content_type: str | None = None) -> None:
        dest = self._prepare_dest(key)
        shutil.copyfile(local_path, dest)

    def upload_bytes(self, data: bytes, key: str, content_type: str | None = None) -> None:
        dest = self._prepare_dest(key)
        dest.write_bytes(data)

    def download(self, key: str, local_path: Path) -> None:
        src = self._resolve(key)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, local_path)

    def read_bytes(self, key: str) -> bytes:
        return self._resolve(key).read_bytes()

    def delete(self, key: str) -> None:
        path = self._resolve(key)
        path.unlink(missing_ok=True)

    def exists(self, key: str) -> bool:
        try:
            return self._resolve(key).exists()
        except PathTraversalError:
            return False

    def get_url(self, key: str) -> str:
        return f"/media/{key}"
