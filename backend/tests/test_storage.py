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


def test_concurrent_uploads_into_new_shared_directory_do_not_race(
    storage: LocalFileStorage, tmp_path: Path
) -> None:
    """Regression: several callers now upload into one `LocalFileStorage`
    concurrently from a bounded thread pool (Roboflow/COCO/CVAT import) —
    e.g. every image of one dataset landing under the same not-yet-created
    `.../images/` directory at once. Before `_prepare_dest`'s lock, that
    raced `Path.resolve()` against a sibling thread's concurrent
    `mkdir(parents=True)` on Windows, occasionally resolving to a path that
    spuriously failed the containment check and raised `PathTraversalError`
    for a perfectly valid key — reproduced directly (~1 in 15 runs of 8
    barrier-synchronized uploads) before the fix."""
    import threading
    import uuid
    from concurrent.futures import ThreadPoolExecutor

    src = tmp_path.parent / "src2.txt"
    src.write_bytes(b"hello")

    for trial in range(15):
        trial_storage = LocalFileStorage(root=tmp_path / f"trial{trial}")
        barrier = threading.Barrier(8)
        errors: list[Exception] = []

        def _upload(i: int) -> None:
            key = f"shared/images/{uuid.uuid4()}.txt"
            barrier.wait()
            try:
                trial_storage.upload(src, key)
            except Exception as exc:  # noqa: BLE001 - collected and asserted below
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(_upload, range(8)))

        assert errors == []
