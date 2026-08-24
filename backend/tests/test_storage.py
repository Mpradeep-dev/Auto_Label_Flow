from __future__ import annotations

from pathlib import Path

import pytest

from app.services.storage.local import LocalFileStorage, PathTraversalError


@pytest.fixture()
def storage(tmp_path: Path) -> LocalFileStorage:
    return LocalFileStorage(root=tmp_path)


def test_upload_and_download_roundtrip(storage: LocalFileStorage, tmp_path: Path) -> None:
    src = tmp_path.parent / "src.txt"
    src.write_bytes(b"hello world")
    storage.upload(src, "a/b/c.txt")
    assert storage.exists("a/b/c.txt")
    assert storage.read_bytes("a/b/c.txt") == b"hello world"


def test_upload_bytes(storage: LocalFileStorage) -> None:
    storage.upload_bytes(b"data", "x/y.bin")
    assert storage.read_bytes("x/y.bin") == b"data"


def test_delete_is_idempotent(storage: LocalFileStorage) -> None:
    storage.upload_bytes(b"data", "z.bin")
    storage.delete("z.bin")
    assert not storage.exists("z.bin")
    storage.delete("z.bin")  # no error on missing file


@pytest.mark.parametrize(
    "malicious_key",
    [
        "../escape.txt",
        "../../etc/passwd",
        "a/../../escape.txt",
        "/etc/passwd",
    ],
)
def test_path_traversal_is_rejected(storage: LocalFileStorage, malicious_key: str) -> None:
    with pytest.raises(PathTraversalError):
        storage.upload_bytes(b"x", malicious_key)


def test_get_url_is_media_relative(storage: LocalFileStorage) -> None:
    assert storage.get_url("a/b.jpg") == "/media/a/b.jpg"
