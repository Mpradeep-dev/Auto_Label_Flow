"""Lets the Datasets page list a connected account's actual Roboflow
projects and versions, instead of asking someone to type a project slug
and version number blind. Read-only — no state written here."""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.services.integrations.roboflow_connect import get_client


def list_projects(db: Session, *, workspace: str | None) -> list[dict]:
    """Projects in `workspace` (or the account's default workspace if
    omitted), from the same workspace payload `connect()` already fetches
    to verify — no per-project API call needed for the list view."""
    rf, config = get_client(db)
    resolved_workspace = workspace or config.get("default_workspace")
    ws = rf.workspace(resolved_workspace)

    projects = []
    for entry in ws.project_list:
        # entry["id"] is "workspace-slug/project-slug" (Roboflow's own
        # convention — see Project.__init__, which parses it the same way).
        project_slug = entry["id"].rsplit("/", 1)[-1]
        projects.append(
            {
                # ws.url, NOT ws.name — name is a display label ("GTPs
                # Workspace", with a space and caps) that Roboflow's own
                # lookups reject; url is the slug ("gtps-workspace") that
                # round-trips correctly into rf.workspace(...) on a later
                # call (list_versions, import, export). Confirmed live
                # against a real connected account: ws.name != ws.url.
                "workspace": ws.url,
                "project": project_slug,
                "name": entry.get("name", project_slug),
                "type": entry.get("type", "unknown"),
                "image_count": entry.get("images", 0),
            }
        )
    return projects


def list_versions(db: Session, *, workspace: str, project_slug: str) -> list[dict]:
    """Versions of one project — a real API call per project.versions(),
    so only fetched once someone drills into a specific project, not for
    every row in the list view."""
    rf, _config = get_client(db)
    project = rf.workspace(workspace).project(project_slug)

    versions = []
    for v in project.versions():
        # v.version is a str (see roboflow.util.versions.unwrap_version_id)
        # even though it's always numeric in practice; Project.version()
        # — which the actual import/export calls use — takes an int, so
        # normalize here rather than pass the mismatch on to the frontend.
        versions.append({"version": int(v.version), "image_count": v.images})
    # Newest first — the version someone most likely wants to import.
    versions.sort(key=lambda v: v["version"], reverse=True)
    return versions


def list_batches(db: Session, *, workspace: str, project_slug: str) -> list[dict]:
    """Upload batches of a project — the groupings Roboflow's own
    Annotate tab splits raw (unversioned) uploads into. Lets the raw
    import path (`import_roboflow_raw_project`) narrow to one batch by
    id instead of always pulling every raw image in the project."""
    rf, _config = get_client(db)
    project = rf.workspace(workspace).project(project_slug)

    payload = project.get_batches()
    # Roboflow's `GET .../batches` wraps the list as {"batches": [...]};
    # tolerate a bare list too in case that ever changes.
    raw_batches = payload.get("batches", []) if isinstance(payload, dict) else payload

    batches = []
    for b in raw_batches or []:
        batch_id = b.get("id")
        if not batch_id:
            continue
        batches.append(
            {
                "id": batch_id,
                "name": b.get("name") or batch_id,
                "image_count": b.get("images") or b.get("numImages") or b.get("image_count") or 0,
            }
        )
    return batches
