"""One-shot diagnostic for the Roboflow export HTTP 500.

Reproduces a single image upload against the same workspace/project the last
EXPORT job used, but prints the raw HTTP status, headers and body that the
`roboflow` SDK swallows (it only re-raises `<Response [500]>`).

Run against the compose stack without rebuilding:

    docker compose exec -T worker python - < backend/scripts/roboflow_upload_probe.py

or, from inside the container:

    python scripts/roboflow_upload_probe.py
"""
from __future__ import annotations

import io
import sys

import app.core.config  # noqa: F401  (sets ULTRALYTICS_SAFE_LOAD before anything else)
import requests
from PIL import Image
from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.roboflow_job import RoboflowJob, RoboflowJobKind
from app.services.integrations.roboflow_connect import get_client


def _mask(key: str) -> str:
    return f"{key[:4]}…{key[-4:]} (len {len(key)})" if key else "<empty>"


def main() -> int:
    db = SessionLocal()

    job = db.scalar(
        select(RoboflowJob)
        .where(RoboflowJob.kind == RoboflowJobKind.EXPORT)
        .order_by(RoboflowJob.created_at.desc())
    )
    if job is None:
        print("No EXPORT RoboflowJob rows found — run an export once first.")
        return 1

    print(f"Latest EXPORT job {job.id}")
    print(f"  status           = {job.status}")
    print(f"  workspace        = {job.workspace!r}")
    print(f"  project_slug     = {job.project_slug!r}")
    print(f"  dataset_version  = {job.dataset_version_id}")
    print(f"  uploaded/failed  = {job.uploaded_count}/{job.failed_count}")
    if job.failures:
        print(f"  first failure    = {job.failures[0]!r}")

    rf, config = get_client(db)
    print(f"\nRoboflow client: workspace(config)={config.get('default_workspace')!r} "
          f"api_key={_mask(config.get('api_key', ''))}")

    import roboflow.config as rfconfig

    api_url = rfconfig.API_URL
    print(f"API_URL = {api_url}")

    # --- resolve the project object the SDK would use -------------------
    try:
        project = rf.workspace(job.workspace).project(job.project_slug)
    except Exception as exc:  # noqa: BLE001
        print(f"\n!! rf.workspace(...).project(...) itself failed: {exc!r}")
        return 1

    interesting = {
        k: getattr(project, k, "<none>")
        for k in ("id", "type", "name", "annotation", "classes", "public", "splits")
    }
    print("\nProject attributes:")
    for k, v in interesting.items():
        print(f"  {k:12} = {v!r}")

    project_url = project.id.rsplit("/")[1] if "/" in project.id else project.id

    # --- raw single upload, full response dump -------------------------
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (127, 127, 127)).save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    upload_url = (
        f"{api_url}/dataset/{project_url}/upload"
        f"?api_key={config['api_key']}&batch=probe-test&name=probe.jpg&split=train"
    )
    print(f"\nPOST {upload_url.replace(config['api_key'], '<key>')}")

    try:
        resp = requests.post(
            upload_url,
            files={"file": ("probe.jpg", img_bytes, "image/jpeg")},
            timeout=(30, 120),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"!! request raised: {exc!r}")
        return 1

    print(f"\n--- RAW RESPONSE ---")
    print(f"status  = {resp.status_code}")
    print(f"headers = {dict(resp.headers)}")
    print(f"body    = {resp.text[:4000]!r}")

    # --- and the SDK path, for the exact exception it raises ----------
    print("\n--- SDK project.upload() path ---")
    try:
        out = project.upload(
            image_path=_tmp_jpeg(img_bytes),
            split="train",
            batch_name="probe-test",
        )
        print(f"ok: {out!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"raised {type(exc).__name__}: {exc!r}")
        code = getattr(exc, "status_code", None)
        if code is not None:
            print(f"  status_code = {code}")

    return 0


def _tmp_jpeg(data: bytes) -> str:
    import tempfile

    f = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    f.write(data)
    f.close()
    return f.name


if __name__ == "__main__":
    sys.exit(main())
