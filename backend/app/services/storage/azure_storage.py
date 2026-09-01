"""Azure Blob Storage ObjectStorage backend. Only imports the
`azure-storage-blob` SDK when actually instantiated, so a local-storage dev
setup never needs the package importable — mirrors minio_storage.py."""
from __future__ import annotations

from pathlib import Path

from app.core.config import settings
from app.services.storage.base import ObjectStorage


class AzureBlobStorage(ObjectStorage):
    def __init__(self) -> None:
        from azure.storage.blob import BlobServiceClient  # local import: keep Azure optional at dev time

        self._service = BlobServiceClient.from_connection_string(
            settings.AZURE_STORAGE_CONNECTION_STRING
        )
        self._container_name = settings.AZURE_STORAGE_CONTAINER
        self._container = self._service.get_container_client(self._container_name)
        if not self._container.exists():
            self._container.create_container()

    def upload(self, local_path: Path, key: str, content_type: str | None = None) -> None:
        from azure.storage.blob import ContentSettings

        content_settings = ContentSettings(content_type=content_type) if content_type else None
        with open(local_path, "rb") as f:
            self._container.upload_blob(
                key, f, overwrite=True, content_settings=content_settings
            )

    def upload_bytes(self, data: bytes, key: str, content_type: str | None = None) -> None:
        from azure.storage.blob import ContentSettings

        content_settings = ContentSettings(content_type=content_type) if content_type else None
        self._container.upload_blob(key, data, overwrite=True, content_settings=content_settings)

    def download(self, key: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob_client = self._container.get_blob_client(key)
        with open(local_path, "wb") as f:
            f.write(blob_client.download_blob().readall())

    def read_bytes(self, key: str) -> bytes:
        return self._container.get_blob_client(key).download_blob().readall()

    def delete(self, key: str) -> None:
        blob_client = self._container.get_blob_client(key)
        if blob_client.exists():
            blob_client.delete_blob()

    def exists(self, key: str) -> bool:
        return self._container.get_blob_client(key).exists()

    def get_url(self, key: str) -> str:
        from datetime import datetime, timedelta, timezone

        from azure.storage.blob import BlobSasPermissions, generate_blob_sas

        blob_client = self._container.get_blob_client(key)
        sas_token = generate_blob_sas(
            account_name=self._service.account_name,
            container_name=self._container_name,
            blob_name=key,
            account_key=self._service.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(hours=6),
        )
        return f"{blob_client.url}?{sas_token}"
