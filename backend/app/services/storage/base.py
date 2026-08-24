"""ObjectStorage abstraction. Local filesystem in dev, MinIO (S3-compatible)
in prod, behind one interface — nothing else in the app should know which
backend is active. `factory.get_storage()` is the only place that decides."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class ObjectStorage(ABC):
    @abstractmethod
    def upload(self, local_path: Path, key: str, content_type: str | None = None) -> None:
        """Upload a local file to `key`."""

    @abstractmethod
    def upload_bytes(self, data: bytes, key: str, content_type: str | None = None) -> None:
        """Upload in-memory bytes to `key`, without a temp file on disk."""

    @abstractmethod
    def download(self, key: str, local_path: Path) -> None:
        """Download `key` to a local file path."""

    @abstractmethod
    def read_bytes(self, key: str) -> bytes:
        """Read `key`'s contents directly into memory."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete the object at `key`. No-op if it doesn't exist."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        ...

    @abstractmethod
    def get_url(self, key: str) -> str:
        """A URL the frontend can load `key` from directly."""
