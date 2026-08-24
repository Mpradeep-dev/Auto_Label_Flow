"""Upload validation shared by every upload endpoint: extension allow-list,
size cap, and safe-filename derivation. Path-traversal protection on storage
KEYS lives in `services/storage/local.py` (it needs filesystem context this
module doesn't have) — this module only handles the HTTP upload boundary."""
from __future__ import annotations

import re
import tempfile
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile, status

from app.core.config import settings

_CHUNK_SIZE = 1024 * 1024  # 1 MiB

_SAFE_STEM = re.compile(r"[^A-Za-z0-9_-]+")


class UploadKind:
    IMAGE = "image"
    VIDEO = "video"


def validate_extension(filename: str, kind: str) -> str:
    """Return the lowercase extension if it's on the allow-list for `kind`, else 400."""
    ext = Path(filename).suffix.lower()
    allowed = (
        settings.ALLOWED_IMAGE_EXTENSIONS if kind == UploadKind.IMAGE else settings.ALLOWED_VIDEO_EXTENSIONS
    )
    if ext not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported {kind} extension {ext!r}. Allowed: {', '.join(allowed)}",
        )
    return ext


async def stream_upload_to_temp(upload: UploadFile, ext: str) -> Path:
    """Stream an UploadFile to a temp file in bounded chunks, enforcing the
    size cap while reading rather than after — an oversized upload is
    rejected without ever buffering the whole thing in memory."""
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    fd, tmp_name = tempfile.mkstemp(suffix=ext)
    tmp_path = Path(tmp_name)
    total = 0
    try:
        with open(fd, "wb") as f:
            while chunk := await upload.read(_CHUNK_SIZE):
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds the {settings.MAX_UPLOAD_SIZE_MB} MB upload limit",
                    )
                f.write(chunk)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    if total == 0:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty upload")
    return tmp_path


def safe_storage_key(*parts: str, original_filename: str) -> str:
    """Build a storage key from trusted path segments (project/dataset ids)
    plus a generated-uuid + sanitized-stem filename — never the raw
    user-supplied filename alone, which could contain path separators or
    traversal sequences."""
    ext = Path(original_filename).suffix.lower()
    stem = _SAFE_STEM.sub("-", Path(original_filename).stem)[:60] or "file"
    unique = uuid.uuid4().hex[:12]
    safe_parts = [re.sub(r"[^A-Za-z0-9_-]+", "-", p) for p in parts]
    return "/".join([*safe_parts, f"{stem}-{unique}{ext}"])
