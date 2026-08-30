"""Single place that decides which ObjectStorage backend is active. Everything
else calls `get_storage()` — nothing else branches on `settings.STORAGE_BACKEND`."""
from __future__ import annotations

from functools import lru_cache

from app.core.config import settings
from app.services.storage.base import ObjectStorage


@lru_cache(maxsize=1)
def get_storage() -> ObjectStorage:
    if settings.STORAGE_BACKEND == "minio":
        from app.services.storage.minio_storage import MinioStorage

        return MinioStorage()

    if settings.STORAGE_BACKEND == "azure":
        from app.services.storage.azure_storage import AzureBlobStorage

        return AzureBlobStorage()

    from app.services.storage.local import LocalFileStorage

    return LocalFileStorage()
