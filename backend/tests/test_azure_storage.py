from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services.storage.azure_storage import AzureBlobStorage


@pytest.fixture()
def mock_container():
    """Patch BlobServiceClient so AzureBlobStorage() never touches real Azure,
    and hand back the mocked container client the code operates on."""
    container = MagicMock()
    container.exists.return_value = True

    service = MagicMock()
    service.get_container_client.return_value = container
    service.account_name = "testaccount"
    service.credential.account_key = "testkey"

    with patch(
        "azure.storage.blob.BlobServiceClient.from_connection_string", return_value=service
    ), patch("app.core.config.settings.AZURE_STORAGE_CONNECTION_STRING", "UseDevelopmentStorage=true"):
        yield container


def test_creates_container_if_missing(mock_container) -> None:
    mock_container.exists.return_value = False
    AzureBlobStorage()
    mock_container.create_container.assert_called_once()


def test_does_not_recreate_existing_container(mock_container) -> None:
    AzureBlobStorage()
    mock_container.create_container.assert_not_called()


def test_upload_bytes_delegates_to_container(mock_container) -> None:
    storage = AzureBlobStorage()
    storage.upload_bytes(b"data", "x/y.bin", content_type="application/octet-stream")
    args, kwargs = mock_container.upload_blob.call_args
    assert args[0] == "x/y.bin"
    assert args[1] == b"data"
    assert kwargs["overwrite"] is True


def test_upload_delegates_to_container(mock_container, tmp_path: Path) -> None:
    src = tmp_path / "src.txt"
    src.write_bytes(b"hello world")
    storage = AzureBlobStorage()
    storage.upload(src, "a/b/c.txt")
    args, kwargs = mock_container.upload_blob.call_args
    assert args[0] == "a/b/c.txt"
    assert kwargs["overwrite"] is True


def test_read_bytes_delegates_to_blob_client(mock_container) -> None:
    blob_client = MagicMock()
    blob_client.download_blob.return_value.readall.return_value = b"content"
    mock_container.get_blob_client.return_value = blob_client

    storage = AzureBlobStorage()
    assert storage.read_bytes("k.bin") == b"content"


def test_delete_is_idempotent(mock_container) -> None:
    blob_client = MagicMock()
    blob_client.exists.return_value = False
    mock_container.get_blob_client.return_value = blob_client

    storage = AzureBlobStorage()
    storage.delete("missing.bin")
    blob_client.delete_blob.assert_not_called()


def test_delete_removes_existing_blob(mock_container) -> None:
    blob_client = MagicMock()
    blob_client.exists.return_value = True
    mock_container.get_blob_client.return_value = blob_client

    storage = AzureBlobStorage()
    storage.delete("k.bin")
    blob_client.delete_blob.assert_called_once()


def test_exists_delegates_to_blob_client(mock_container) -> None:
    blob_client = MagicMock()
    blob_client.exists.return_value = True
    mock_container.get_blob_client.return_value = blob_client

    storage = AzureBlobStorage()
    assert storage.exists("k.bin") is True


def test_get_url_returns_sas_signed_url(mock_container) -> None:
    blob_client = MagicMock()
    blob_client.url = "https://testaccount.blob.core.windows.net/annotate/k.bin"
    mock_container.get_blob_client.return_value = blob_client

    storage = AzureBlobStorage()
    with patch(
        "azure.storage.blob.generate_blob_sas", return_value="sig=abc123"
    ) as mock_sas:
        url = storage.get_url("k.bin")

    assert url == "https://testaccount.blob.core.windows.net/annotate/k.bin?sig=abc123"
    assert mock_sas.call_args.kwargs["blob_name"] == "k.bin"
    assert mock_sas.call_args.kwargs["account_name"] == "testaccount"
