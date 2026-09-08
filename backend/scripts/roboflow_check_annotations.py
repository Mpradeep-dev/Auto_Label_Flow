"""Check whether images this app pushed to Roboflow actually landed
annotated, or show as "Unannotated" (issues #12 / #16). Looks up the
`autolabelflow-*` batch(es) via the SDK's own /batches endpoint, then
queries /search for each batch's images and reports each one's real
annotation count as Roboflow sees it.

Run: docker compose exec worker python roboflow_check_annotations.py
"""
from __future__ import annotations

import sys

import app.core.config  # noqa: F401
import requests

from app.db.session import SessionLocal
from app.models.roboflow_job import RoboflowJob, RoboflowJobKind
from app.services.integrations.roboflow_connect import get_client
from sqlalchemy import select


def main() -> int:
    db = SessionLocal()
    job = db.scalar(
        select(RoboflowJob)
        .where(RoboflowJob.kind == RoboflowJobKind.EXPORT, RoboflowJob.status == "COMPLETED")
        .order_by(RoboflowJob.created_at.desc())
    )
    if job is None:
        print("No COMPLETED export job found.")
        return 1
    print(f"job={job.id} workspace={job.workspace} project={job.project_slug} "
          f"uploaded={job.uploaded_count} failed={job.failed_count}")

    rf, config = get_client(db)
    project = rf.workspace(job.workspace).project(job.project_slug)
    batches = project.get_batches().get("batches", [])
    autolabel_batches = [b for b in batches if "autolabelflow" in b.get("name", "").lower()]
    print(f"\nAutoLabelFlow batches on Roboflow: {[(b['name'], b.get('images')) for b in autolabel_batches]}")
    if not autolabel_batches:
        print("No autolabelflow batch found on Roboflow side.")
        return 1

    import roboflow.config as rfconfig

    api_key = config["api_key"]
    for b in autolabel_batches:
        batch_id = b["id"]
        url = f"{rfconfig.API_URL}/{project.id}/search?api_key={api_key}"
        payload = {
            "offset": 0,
            "limit": 200,
            "batch": True,
            "batch_id": batch_id,
            "fields": ["id", "name", "annotations"],
        }
        resp = requests.post(url, json=payload, timeout=30)
        print(f"\nBatch {b['name']!r} ({batch_id}) -> HTTP {resp.status_code}")
        try:
            body = resp.json()
        except Exception:
            print(f"  non-JSON body: {resp.text[:500]!r}")
            continue
        if "results" not in body:
            print(f"  raw body: {body}")
            continue
        results = body.get("results", [])
        annotated = 0
        unannotated = []
        for item in results:
            ann = item.get("annotations") or {}
            count = ann.get("count") if isinstance(ann, dict) else None
            if count:
                annotated += 1
            else:
                unannotated.append(item.get("name"))
        print(f"  {len(results)} images returned")
        print(f"  annotated:   {annotated}")
        print(f"  unannotated: {len(unannotated)}  {unannotated[:10]}")
        if results:
            print(f"  sample raw item: {results[0]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
