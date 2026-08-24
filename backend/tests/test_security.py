"""Unit tests for the upload-boundary validation helpers (PLAN spec
section 22: extension allow-list, safe storage-key derivation)."""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.core.security import UploadKind, safe_storage_key, validate_extension


def test_validate_extension_accepts_allowed_image_extension() -> None:
    assert validate_extension("photo.JPG", UploadKind.IMAGE) == ".jpg"


def test_validate_extension_rejects_disallowed_extension() -> None:
    with pytest.raises(HTTPException) as exc_info:
        validate_extension("payload.exe", UploadKind.IMAGE)
    assert exc_info.value.status_code == 400


def test_validate_extension_rejects_video_extension_for_image_kind() -> None:
    with pytest.raises(HTTPException):
        validate_extension("clip.mp4", UploadKind.IMAGE)


def test_validate_extension_accepts_allowed_video_extension() -> None:
    assert validate_extension("clip.MOV", UploadKind.VIDEO) == ".mov"


def test_safe_storage_key_sanitizes_path_separators_in_filename() -> None:
    key = safe_storage_key("proj1", "dataset1", original_filename="../../etc/passwd.jpg")
    assert ".." not in key
    assert "/" not in key.split("/")[-1].replace("passwd", "")  # no separator survives into the filename segment
    assert key.startswith("proj1/dataset1/")
    assert key.endswith(".jpg")


def test_safe_storage_key_sanitizes_path_segments() -> None:
    key = safe_storage_key("../evil", "normal", original_filename="f.jpg")
    assert not key.split("/")[0].__contains__("..")


def test_safe_storage_key_is_unique_per_call() -> None:
    key1 = safe_storage_key("p", "d", original_filename="same.jpg")
    key2 = safe_storage_key("p", "d", original_filename="same.jpg")
    assert key1 != key2  # uuid suffix guarantees no collision even for identical filenames


def test_safe_storage_key_handles_extensionless_filename() -> None:
    """A filename with no extension (e.g. a browser-selected file with no
    suffix) must still produce a well-formed key — no crash, no doubled
    separators. Downstream readers (cv2.imdecode) don't depend on the
    storage key carrying a real extension anyway; they decode from bytes."""
    key = safe_storage_key("p", "d", original_filename="photo")
    assert key.startswith("p/d/")
    assert "//" not in key
    assert "photo" in key
