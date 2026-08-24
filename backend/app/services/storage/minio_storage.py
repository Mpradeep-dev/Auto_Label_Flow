"""MinIO (S3-compatible) ObjectStorage — the prod-mode backend. Only
imports the `minio` SDK when actually instantiated, so a local-storage dev
setup never needs the package importable."""
from __future__ import annotations

import io
from pathlib import Path

from app.core.config import settings
from app.services.storage.base import ObjectStorage


class MinioStorage(ObjectStorage):
    def __init__(self) -> None:
        from minio import Minio  # local import: keep MinIO optional at dev time

        self._client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
            region=settings.MINIO_REGION,
        )
        self._bucket = settings.MINIO_BUCKET
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)

        # A second client, used only for presigned_get_object: it must sign
        # URLs against the host a *browser* can reach, which differs from
        # MINIO_ENDPOINT inside Docker Compose (see config.py). Reuses the
        # same MinIO instance/credentials — only the host in the signed URL
        # changes. `region` is passed explicitly on both clients so signing
        # never triggers its own GetBucketLocation network call against
        # that (possibly container-unreachable) public endpoint.
        if settings.minio_public_endpoint == settings.MINIO_ENDPOINT:
            self._public_client = self._client
        else:
            self._public_client = Minio(
                settings.minio_public_endpoint,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_SECURE,
                region=settings.MINIO_REGION,
            )

    def upload(self, local_path: Path, key: str, content_type: str | None = None) -> None:
        self._client.fput_object(self._bucket, key, str(local_path), content_type=content_type)

    def upload_bytes(self, data: bytes, key: str, content_type: str | None = None) -> None:
        self._client.put_object(
            self._bucket, key, io.BytesIO(data), length=len(data), content_type=content_type
        )

    def download(self, key: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._client.fget_object(self._bucket, key, str(local_path))

    def read_bytes(self, key: str) -> bytes:
        response = self._client.get_object(self._bucket, key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def delete(self, key: str) -> None:
        self._client.remove_object(self._bucket, key)

    def exists(self, key: str) -> bool:
        from minio.error import S3Error

        try:
            self._client.stat_object(self._bucket, key)
            return True
        except S3Error:
            return False

    def get_url(self, key: str) -> str:
        from datetime import timedelta

        return self._public_client.presigned_get_object(self._bucket, key, expires=timedelta(hours=6))
