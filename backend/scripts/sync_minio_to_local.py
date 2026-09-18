"""One-shot data-recovery tool: copy every object that exists in MinIO but
not on local disk into `LOCAL_STORAGE_DIR`.

Why this is needed: this repo's local dev setup lets you run the backend
either fully via `docker compose up --build` (STORAGE_BACKEND=minio) or
natively against `backend/venv` (STORAGE_BACKEND=local, per backend/.env) —
both against the *same* Postgres database (see AGENTS.md's two profiles).
Switching between the two after images/videos/exports were already created
leaves the DB rows (storage_key) pointing at bytes that only exist under
whichever backend was active at the time. Reading an image created while
running one way, from a process running the other way, 404s/FileNotFoundErrors
even though the data isn't actually lost — it's just in the other backend.

This script makes the union available locally by copying (never deleting)
every object found in MinIO's bucket down to local disk, skipping whatever
already exists there. Safe to re-run.

Usage (repo root, backend venv active, MinIO reachable at localhost:9000 —
i.e. `docker compose up -d minio`):

    cd backend
    ./venv/Scripts/python scripts/sync_minio_to_local.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys

import app.core.config  # noqa: F401  (sets ULTRALYTICS_SAFE_LOAD before anything else)

from app.core.config import settings
from app.services.storage.local import LocalFileStorage
from app.services.storage.minio_storage import MinioStorage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="List what would be copied without writing anything"
    )
    args = parser.parse_args()

    from minio import Minio

    minio = MinioStorage()
    local = LocalFileStorage()
    # `ObjectStorage` has no listing method (nothing else in the app needs
    # one), so this one-off recovery tool talks to the raw SDK directly for
    # enumeration only — every actual read/write still goes through the two
    # `ObjectStorage` implementations above, config-driven like the rest of
    # the app.
    raw_client = Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
        region=settings.MINIO_REGION,
    )

    print(f"MinIO bucket   : {settings.MINIO_BUCKET} @ {settings.MINIO_ENDPOINT}")
    print(f"Local root     : {local.root}")
    print(f"Mode           : {'dry-run (no writes)' if args.dry_run else 'copy'}")
    print()

    copied = 0
    already_local = 0
    failed: list[str] = []

    for obj in raw_client.list_objects(settings.MINIO_BUCKET, recursive=True):
        object_name = obj.object_name
        if local.exists(object_name):
            already_local += 1
            continue

        if args.dry_run:
            print(f"WOULD COPY  {object_name}")
            copied += 1
            continue

        try:
            data = minio.read_bytes(object_name)
            local.upload_bytes(data, object_name)
            copied += 1
            print(f"copied      {object_name} ({len(data)} bytes)")
        except Exception as exc:  # noqa: BLE001 — keep going, report at the end
            failed.append(f"{object_name}: {exc}")
            print(f"FAILED      {object_name}: {exc}")

    print()
    print(f"Already local : {already_local}")
    print(f"Copied        : {copied}")
    print(f"Failed        : {len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
