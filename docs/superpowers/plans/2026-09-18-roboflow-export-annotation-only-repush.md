# Roboflow Export: Annotation-Only Re-Push Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the real Roboflow export workflow (Roboflow assigns images → we pull them in → developer annotates → we push back) so that re-pushing an image Roboflow already has (a) actually updates its annotation instead of silently no-opping, (b) never re-uploads image bytes when we already know the image's Roboflow id, and (c) reports honestly (new image vs. existing-image update vs. failure) instead of the current "all skipped as duplicates" message that reads like nothing happened even when new images *did* get pushed, and hides whether re-annotated existing images landed at all.

**Architecture:** Two layered fixes, in this order:
1. **Surgical fix (works for every import path, no schema dependency):** `Project.upload()`'s own duplicate-detection already resolves the pre-existing image's id and calls `save_annotation()` for us — but our code never passes `annotation_overwrite=True`, so Roboflow's 409 "already annotated" response makes that update silently no-op. Adding the flag, plus replacing the current `uploaded`/`duplicates` counters with a three-way `new_image_uploaded` / `annotation_updated` / `unchanged` outcome per image, fixes correctness and reporting for every export regardless of where the image came from.
2. **Direct routing (raw-pull imports only):** persist each raw-pull-imported image's Roboflow id (`import_roboflow_raw_project` already fetches it per item, just never stores it) and, on export, skip `Project.upload()` entirely for images we already have a matching id for — call `Project.save_annotation()` directly. No re-upload of unchanged bytes, no dependency on Roboflow's dedup at all. Falls back to the normal upload path on a stale id (404) or a workspace/project mismatch. **Versioned imports** (`import_roboflow_project`, pulling a generated Version as a YOLO zip) never see a per-image Roboflow id and are intentionally left on fix #1 only — see Task 3's docstring note.

**Tech Stack:** FastAPI + SQLAlchemy + Alembic (backend), the installed `roboflow` Python SDK (`backend/venv/Lib/site-packages/roboflow`, traced directly — see Global Constraints), Celery (export runs as a background job), React/TypeScript (frontend, type-only change).

**Spec:** This plan *is* the spec — written directly against the traced SDK behavior and the existing codebase, per the requirements the user gave in-conversation (persist `roboflow_image_id`, route export on it, never let duplicates block new images, report `new_image_uploaded`/`annotation_updated`/`failed` honestly). No separate spec doc exists.

## Global Constraints

- **Traced, not guessed, against the exact installed SDK** at `backend/venv/Lib/site-packages/roboflow` (roboflow-python). Every claim below about SDK behavior is backed by a specific file:line cited in the task it's used in. Do not "improve" on these based on the SDK's public docs or memory — the installed version is what runs.
- `annotation_overwrite=True` must be passed on **every** annotation write this plan touches (both the `Project.upload()` fallback path and the new direct `Project.save_annotation()` path) — see Task 4 for why: the SDK default (`False`) makes Roboflow's 409 "already annotated" response a silent no-op.
- `Project.save_annotation()`'s `annotation_labelmap` parameter must be the **loaded** `{index: name}` dict (`roboflow.util.image_utils.load_labelmap(path)`), never a bare file path — `Project.upload()`'s internal `single_upload()` converts it before calling `save_annotation` (`roboflow/core/project.py:604-605`); calling `save_annotation()` directly, as Task 5 does, means doing that conversion ourselves.
- Roboflow's `batch` parameter (and therefore this app's `batch_name`) must match `^[a-z0-9_-]{1,64}$` — already handled by `_sanitize_batch_name`; nothing in this plan changes that.
- New Alembic migrations chain off the current head (`d1e5a7c3f902`) — confirmed via `grep -r "down_revision.*d1e5a7c3f902" backend/alembic/versions` returning nothing.
- New DB columns are nullable with no default query burden — `Image.roboflow_image_id`/`roboflow_workspace`/`roboflow_project_slug` are `String(200)`, matching `RoboflowJob.workspace`/`project_slug`'s existing convention exactly (`app/models/roboflow_job.py:64-65`).
- Existing tests in `backend/tests/test_roboflow_jobs.py` that assert the *old* return shape of `push_version_to_roboflow` (4-tuple) or assume `note is None` on a clean mixed push **will** break and must be updated in the task that changes that behavior, not left broken.
- Backend tests run via the project's normal pytest invocation (see `AGENTS.md` for the exact command) against the `real_db_session`/`real_client` fixtures already used throughout `test_roboflow_jobs.py` — these hit a real (test-schema) Postgres connection with FK constraints enforced, so any test creating an `Image` row directly must first create a real `Project` and `Dataset` row (no shortcut around FKs).

---

## File Structure

- `backend/app/models/image.py` — **modify**: add `roboflow_image_id`, `roboflow_workspace`, `roboflow_project_slug` columns.
- `backend/alembic/versions/a4f1c8e2b567_image_roboflow_provenance.py` — **create**: migration for the above.
- `backend/app/models/roboflow_job.py` — **modify**: add `new_images_count`, `annotations_updated_count` columns.
- `backend/alembic/versions/b8e3d1a94f72_roboflow_job_new_images_and_annotations_updated_counts.py` — **create**: migration for the above.
- `backend/app/schemas/integration.py` — **modify**: `RoboflowJobRead` gains the two new fields.
- `backend/app/services/integrations/roboflow_import.py` — **modify**: `import_roboflow_raw_project` persists the Roboflow image id/workspace/project_slug it already fetches per item.
- `backend/app/services/integrations/roboflow_export.py` — **modify**: the bulk of the logic — outcome tracking, the `annotation_overwrite=True` fix, the direct-routing path, honest note-building.
- `backend/app/workers/tasks/roboflow.py` — **modify**: `run_roboflow_export` adapts to the new `PushResult` return type and sets the two new job columns.
- `backend/tests/test_roboflow_jobs.py` — **modify**: update tests broken by the return-shape/behavior change, add new coverage for the direct-routing path.
- `frontend/src/types/index.ts` — **modify**: `RoboflowJob` interface gains the two new optional-display fields.

---

### Task 1: Persist Roboflow image provenance on `Image`

**Files:**
- Modify: `backend/app/models/image.py`
- Create: `backend/alembic/versions/a4f1c8e2b567_image_roboflow_provenance.py`
- Test: `backend/tests/test_roboflow_jobs.py` (new small model-level test)

**Interfaces:**
- Produces: `Image.roboflow_image_id: str | None`, `Image.roboflow_workspace: str | None`, `Image.roboflow_project_slug: str | None` — consumed by Task 3 (import, writes these) and Task 5 (export, reads these).

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_roboflow_jobs.py` (anywhere after the imports, e.g. right before `def test_roboflow_import_job_completes_and_creates_dataset`):

```python
def test_image_model_has_roboflow_provenance_columns(real_db_session, unique_name: str) -> None:
    """New columns exist and round-trip — the foundation Task 3 (import)
    writes to and Task 5 (export) reads from."""
    from app.models.image import Image, ImageSourceType
    from app.models.project import Project
    from app.models.dataset import Dataset

    project = Project(name=unique_name, slug=unique_name, class_config=[{"id": 0, "name": "cone"}])
    real_db_session.add(project)
    real_db_session.flush()
    dataset = Dataset(project_id=project.id, name="ds")
    real_db_session.add(dataset)
    real_db_session.flush()

    image = Image(
        project_id=project.id,
        dataset_id=dataset.id,
        storage_key="k",
        original_filename="a.jpg",
        width=64,
        height=48,
        source_type=ImageSourceType.UPLOAD,
        roboflow_image_id="rf-abc123",
        roboflow_workspace="my-workspace",
        roboflow_project_slug="cones",
    )
    real_db_session.add(image)
    real_db_session.commit()
    real_db_session.refresh(image)

    assert image.roboflow_image_id == "rf-abc123"
    assert image.roboflow_workspace == "my-workspace"
    assert image.roboflow_project_slug == "cones"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_roboflow_jobs.py::test_image_model_has_roboflow_provenance_columns -v`
Expected: FAIL — `TypeError: 'roboflow_image_id' is an invalid keyword argument for Image` (column doesn't exist yet).

- [ ] **Step 3: Add the columns to the model**

In `backend/app/models/image.py`, add after the `review_status` block (after line 68, before the `difficulty_score` comment block):

```python
    # Set only when this image was pulled in from a Roboflow *raw* (no
    # Version yet) pull (`import_roboflow_raw_project`, which fetches each
    # item's own Roboflow id already — see its `_stage_raw_image`).
    # Versioned imports (`import_roboflow_project`, a downloaded YOLO zip)
    # never see a per-image Roboflow id and leave these NULL. Export
    # (`roboflow_export.py`) uses these — when set AND matching the export
    # target workspace/project — to update the image's annotation directly
    # via `Project.save_annotation()` instead of re-uploading its
    # (unchanged) bytes through `Project.upload()` and relying on
    # Roboflow's own byte-content duplicate detection.
    roboflow_image_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    roboflow_workspace: Mapped[str | None] = mapped_column(String(200), nullable=True)
    roboflow_project_slug: Mapped[str | None] = mapped_column(String(200), nullable=True)
```

- [ ] **Step 4: Write the migration**

Create `backend/alembic/versions/a4f1c8e2b567_image_roboflow_provenance.py`:

```python
"""image roboflow provenance (image_id/workspace/project_slug) for annotation-only re-push

Revision ID: a4f1c8e2b567
Revises: d1e5a7c3f902
Create Date: 2026-09-18 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a4f1c8e2b567'
down_revision: Union[str, None] = 'd1e5a7c3f902'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('images', sa.Column('roboflow_image_id', sa.String(length=200), nullable=True))
    op.add_column('images', sa.Column('roboflow_workspace', sa.String(length=200), nullable=True))
    op.add_column('images', sa.Column('roboflow_project_slug', sa.String(length=200), nullable=True))


def downgrade() -> None:
    op.drop_column('images', 'roboflow_project_slug')
    op.drop_column('images', 'roboflow_workspace')
    op.drop_column('images', 'roboflow_image_id')
```

- [ ] **Step 5: Apply the migration and run the test**

Run: `cd backend && alembic upgrade head`
Then: `pytest backend/tests/test_roboflow_jobs.py::test_image_model_has_roboflow_provenance_columns -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/models/image.py backend/alembic/versions/a4f1c8e2b567_image_roboflow_provenance.py backend/tests/test_roboflow_jobs.py
git commit -m "feat(roboflow): persist per-image Roboflow provenance for annotation-only re-push"
```

---

### Task 2: `RoboflowJob` gets `new_images_count` / `annotations_updated_count`

**Files:**
- Modify: `backend/app/models/roboflow_job.py`
- Modify: `backend/app/schemas/integration.py`
- Create: `backend/alembic/versions/b8e3d1a94f72_roboflow_job_new_images_and_annotations_updated_counts.py`
- Test: `backend/tests/test_roboflow_jobs.py`

**Interfaces:**
- Produces: `RoboflowJob.new_images_count: int`, `RoboflowJob.annotations_updated_count: int` (both `nullable=False, default=0`), and the matching fields on `RoboflowJobRead` — consumed by Task 6 (worker task sets them) and the frontend (Task 7, read-only display).
- `RoboflowJob.uploaded_count` keeps its existing meaning ("total images that landed successfully on Roboflow, whatever kind of landing") — Task 6 sets it to `new_images_count + annotations_updated_count + unchanged`, so no existing consumer of `uploaded_count` breaks.

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_roboflow_jobs.py`:

```python
def test_roboflow_job_model_has_new_images_and_annotations_updated_counts(
    real_db_session, unique_name: str
) -> None:
    from app.models.roboflow_job import RoboflowJob, RoboflowJobKind
    from app.models.project import Project

    # `project_id` is a real FK (Postgres enforces it in this test env, same
    # as the `RoboflowJob(...)` constructions already in this file around
    # lines 826 and 861 — both use a real project id from a created
    # project, never a bare random uuid), so a real Project row comes first.
    project = Project(name=unique_name, slug=unique_name, class_config=[{"id": 0, "name": "cone"}])
    real_db_session.add(project)
    real_db_session.commit()

    job = RoboflowJob(
        project_id=project.id,
        kind=RoboflowJobKind.EXPORT,
        workspace="ws",
        project_slug="proj",
    )
    real_db_session.add(job)
    real_db_session.commit()
    real_db_session.refresh(job)

    assert job.new_images_count == 0
    assert job.annotations_updated_count == 0

    job.new_images_count = 3
    job.annotations_updated_count = 7
    real_db_session.commit()
    real_db_session.refresh(job)
    assert (job.new_images_count, job.annotations_updated_count) == (3, 7)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_roboflow_jobs.py::test_roboflow_job_model_has_new_images_and_annotations_updated_counts -v`
Expected: FAIL — `AttributeError: 'RoboflowJob' object has no attribute 'new_images_count'` on the `assert job.new_images_count == 0` line (the columns don't exist yet).

- [ ] **Step 3: Add the columns to the model**

In `backend/app/models/roboflow_job.py`, right after `uploaded_count`/`failed_count`/`failures` (after line 103):

```python
    # Split out of `uploaded_count` (which stays "everything that landed on
    # Roboflow, whatever kind of landing") so the export result/UI can say
    # which of that total was a genuinely new image vs. an existing
    # Roboflow image (this app already knew about, or Roboflow's own
    # duplicate-detection resolved) whose annotation was written/replaced.
    # See `push_version_to_roboflow`'s `PushResult` in roboflow_export.py.
    new_images_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    annotations_updated_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
```

- [ ] **Step 4: Add the fields to the read schema**

In `backend/app/schemas/integration.py`, in `RoboflowJobRead` (after `uploaded_count: int` and before `failed_count: int`, around line 157):

```python
    uploaded_count: int
    new_images_count: int
    annotations_updated_count: int
    failed_count: int
```

- [ ] **Step 5: Write the migration**

Create `backend/alembic/versions/b8e3d1a94f72_roboflow_job_new_images_and_annotations_updated_counts.py`:

```python
"""roboflow job new_images_count / annotations_updated_count (export: honest new-vs-updated reporting)

Revision ID: b8e3d1a94f72
Revises: a4f1c8e2b567
Create Date: 2026-09-18 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b8e3d1a94f72'
down_revision: Union[str, None] = 'a4f1c8e2b567'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('roboflow_jobs', sa.Column('new_images_count', sa.Integer(), nullable=False, server_default='0'))
    op.add_column(
        'roboflow_jobs', sa.Column('annotations_updated_count', sa.Integer(), nullable=False, server_default='0')
    )
    op.alter_column('roboflow_jobs', 'new_images_count', server_default=None)
    op.alter_column('roboflow_jobs', 'annotations_updated_count', server_default=None)


def downgrade() -> None:
    op.drop_column('roboflow_jobs', 'annotations_updated_count')
    op.drop_column('roboflow_jobs', 'new_images_count')
```

- [ ] **Step 6: Apply the migration and run the test**

Run: `cd backend && alembic upgrade head`
Then: `pytest backend/tests/test_roboflow_jobs.py::test_roboflow_job_model_has_new_images_and_annotations_updated_counts -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add backend/app/models/roboflow_job.py backend/app/schemas/integration.py backend/alembic/versions/b8e3d1a94f72_roboflow_job_new_images_and_annotations_updated_counts.py backend/tests/test_roboflow_jobs.py
git commit -m "feat(roboflow): add new_images_count/annotations_updated_count to RoboflowJob"
```

---

### Task 3: Persist the Roboflow image id on raw-pull import

**Files:**
- Modify: `backend/app/services/integrations/roboflow_import.py:647-698` (the `_stage_raw_image`/`Image(...)` construction inside `import_roboflow_raw_project`)
- Test: `backend/tests/test_roboflow_jobs.py`

**Interfaces:**
- Consumes: `Image.roboflow_image_id`/`roboflow_workspace`/`roboflow_project_slug` (Task 1). `item["id"]` — already present on every raw-pull item dict (`roboflow_import.py:653`: `_rf_image_details(rf_project, config["api_key"], item["id"])`), just never stored.
- Produces: every `Image` row created by `import_roboflow_raw_project` now carries its source Roboflow id — consumed by Task 5's export routing.

**Important — scope note (do not extend this to `import_roboflow_project`):** the versioned-import path downloads a generated YOLO zip (`_download_version_dataset` → `rf_version.download(...)`, `roboflow_import.py:196-225`) whose image filenames are Roboflow's own export-generated names (e.g. `foo_jpg.rf.<hash>.jpg`), not IDs resolvable against `/images/<id>`. Getting a per-image id there would need an extra `project.search()`/match-by-filename call per image — out of scope per the user's agreed scoping ("Surgical fix + Part B for raw-pull imports only"). Leave `import_roboflow_project`'s `Image(...)` construction (`roboflow_import.py:360-368`) untouched; those images simply keep `roboflow_image_id = NULL` and fall through to the Task 4 surgical-fix path on export, same as before this plan.

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_roboflow_jobs.py`, near `test_roboflow_import_job_raw_pull_when_no_version` (reuses the same `_fake_get`/`_fake_search_post` fixtures already defined at the top of the file):

```python
def test_roboflow_raw_pull_import_persists_roboflow_image_id(
    connected_roboflow: TestClient, monkeypatch, unique_name: str
) -> None:
    """The raw-pull path already fetches each item's Roboflow id
    (`_rf_image_details(..., item["id"])`) — it must now also store it, so
    export can route these images through `save_annotation()` directly
    instead of re-uploading them (Task 5)."""
    import app.services.integrations.roboflow_import as roboflow_import_module

    monkeypatch.setattr(roboflow_import_module.requests, "get", _fake_get)
    monkeypatch.setattr(roboflow_import_module.requests, "post", _fake_search_post)

    project_id = connected_roboflow.post("/api/v1/projects", json={"name": unique_name}).json()["id"]

    resp = connected_roboflow.post(
        f"/api/v1/projects/{project_id}/import/roboflow",
        json={"workspace": "my-workspace", "project": "ground"},
    )
    assert resp.status_code == 202, resp.text
    job = resp.json()
    assert job["status"] == "COMPLETED"

    dataset_id = job["result_dataset_id"]
    images = connected_roboflow.get(f"/api/v1/datasets/{dataset_id}/images").json()["items"]
    assert len(images) == 2

    from app.db.session import SessionLocal
    from app.models.image import Image

    db = SessionLocal()
    try:
        rows = db.query(Image).filter(Image.dataset_id == dataset_id).all()
        ids = {r.roboflow_image_id for r in rows}
        assert ids == {"raw-img-1", "raw-img-2"}
        assert all(r.roboflow_workspace == "my-workspace" for r in rows)
        assert all(r.roboflow_project_slug == "ground" for r in rows)
    finally:
        db.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_roboflow_jobs.py::test_roboflow_raw_pull_import_persists_roboflow_image_id -v`
Expected: FAIL — `assert ids == {"raw-img-1", "raw-img-2"}` fails because `ids == {None}` (the columns exist from Task 1 but nothing writes to them yet).

- [ ] **Step 3: Implement**

In `backend/app/services/integrations/roboflow_import.py`, inside `import_roboflow_raw_project`'s `_stage_raw_image` (around line 672), the current return is:

```python
        return {
            "key": key,
            "original_filename": original_filename,
            "width": width,
            "height": height,
            "annotation": details.get("annotation") or {},
        }
```

Change to also carry the item's Roboflow id:

```python
        return {
            "key": key,
            "original_filename": original_filename,
            "width": width,
            "height": height,
            "annotation": details.get("annotation") or {},
            "roboflow_image_id": item["id"],
        }
```

Then the `Image(...)` construction a few lines below (around line 688):

```python
            image = Image(
                project_id=project_id,
                dataset_id=dataset.id,
                storage_key=result["key"],
                original_filename=result["original_filename"],
                width=result["width"],
                height=result["height"],
                source_type=ImageSourceType.UPLOAD,
            )
```

becomes:

```python
            image = Image(
                project_id=project_id,
                dataset_id=dataset.id,
                storage_key=result["key"],
                original_filename=result["original_filename"],
                width=result["width"],
                height=result["height"],
                source_type=ImageSourceType.UPLOAD,
                roboflow_image_id=result["roboflow_image_id"],
                roboflow_workspace=workspace,
                roboflow_project_slug=project_slug,
            )
```

(`workspace`/`project_slug` are already `import_roboflow_raw_project`'s own parameters — no new plumbing needed.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_roboflow_jobs.py::test_roboflow_raw_pull_import_persists_roboflow_image_id -v`
Expected: PASS

- [ ] **Step 5: Run the full existing import test suite to check for regressions**

Run: `pytest backend/tests/test_roboflow_jobs.py -k import -v`
Expected: all PASS (this change is additive to the `Image(...)` call, nothing existing reads/asserts on these fields being absent).

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/integrations/roboflow_import.py backend/tests/test_roboflow_jobs.py
git commit -m "feat(roboflow): persist source Roboflow image id on raw-pull import"
```

---

### Task 4: Surgical fix — `annotation_overwrite=True` + honest three-way outcome tracking

**Files:**
- Modify: `backend/app/services/integrations/roboflow_export.py`
- Modify: `backend/app/workers/tasks/roboflow.py`
- Test: `backend/tests/test_roboflow_jobs.py`

**Interfaces:**
- Consumes: nothing new yet (Task 5 adds the id-routing on top of this).
- Produces: `_PushOutcome` (enum: `NEW_IMAGE`, `ANNOTATION_UPDATED`, `UNCHANGED`), `PushResult` (`NamedTuple`: `new_images: int, annotations_updated: int, unchanged: int, failed: int, failures: list[str], note: str | None`) — `push_version_to_roboflow` now returns `PushResult` instead of a 4-tuple. `_upload_one_image(project, *, has_annotation: bool, **upload_kwargs) -> _PushOutcome` (signature changed from `-> bool`). `_build_export_note(*, new_images: int, annotations_updated: int, unchanged: int, review_job_note: str | None) -> str | None`.

**Why `annotation_overwrite=True` matters (traced, not guessed):** `Project.upload()` → `single_upload()` (`roboflow/core/project.py:607-636`) resolves `image_id = uploaded_image["id"]` from the upload response *even when Roboflow reports it a duplicate* (confirmed against the SDK's own bulk-upload logging at `roboflow/core/workspace.py:611-627`, which explicitly logs `annotations = OK/WARN/ERR` for a `[DUPLICATE]` image), then still calls `self.save_annotation(...)`. But `single_upload`'s `annotation_overwrite` parameter defaults to `False`, and this app's `_upload_kwargs()` never sets it — so `rfapi.save_annotation` (`roboflow/adapters/rfapi.py:815-821`) hits Roboflow's HTTP 409 "already annotated" response and returns `{"warn": "already annotated"}` **without writing anything**. Every image that already has *any* annotation in Roboflow (the ground truth it was imported with, or a prediction from an earlier push) silently keeps its old annotation forever, no matter how many times you re-push. This is the actual root cause of "my re-annotated labels never show up in Roboflow."

- [ ] **Step 1: Write the failing tests (update the two existing duplicate tests to the new return shape + real behavior)**

Replace `test_push_version_all_duplicates_skips_batch_lookup_with_clear_note` in `backend/tests/test_roboflow_jobs.py` (currently lines 1675-1727) with:

```python
def test_push_version_all_duplicates_updates_annotations_not_skipped(real_db_session, monkeypatch) -> None:
    """Roboflow reports every upload as a duplicate (byte-identical image
    already in the project — this app's own normal
    import-then-annotate-then-push workflow). That must NOT read as
    nothing happened: each duplicate's annotation is still written (via
    the SDK's own image-id-from-duplicate-response + save_annotation, now
    with overwrite=True so it isn't silently rejected), counted as
    `annotations_updated`, and the note says so plainly — no batch lookup,
    since nothing new landed to file a review job for."""
    import uuid as _uuid

    from app.services.integrations import roboflow_export as mod

    def _fake_write_yolo_dataset(db, *, version_id, root):
        for split in ("train", "valid", "test"):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
        for i in range(3):
            (root / "images" / "train" / f"img{i}.jpg").write_bytes(_jpeg_bytes())
            (root / "labels" / "train" / f"img{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (root / "data.yaml").write_text("names: ['cone']\n", encoding="utf-8")
        return root / "data.yaml"

    monkeypatch.setattr(mod, "write_yolo_dataset", _fake_write_yolo_dataset)

    get_batches_calls = []
    upload_calls = []

    class _Proj:
        def upload(self, **kwargs):
            upload_calls.append(kwargs)
            return [{"image": {"id": "existing-image-id", "success": False, "duplicate": True}}]

        def get_batches(self):
            get_batches_calls.append(1)
            return {"batches": []}

    class _WS:
        def project(self, slug):
            return _Proj()

    class _RF:
        def workspace(self, ws):
            return _WS()

    monkeypatch.setattr(mod, "get_client", lambda db: (_RF(), {"default_labeler_email": "a@b.com"}))

    result = mod.push_version_to_roboflow(
        real_db_session, version_id=_uuid.uuid4(), workspace="ws", project_slug="proj"
    )

    assert (result.new_images, result.annotations_updated, result.unchanged, result.failed) == (0, 3, 0, 0)
    assert not get_batches_calls  # never searched for a batch that was never created
    assert result.note is not None
    assert "3 existing image" in result.note
    assert "updated" in result.note
    assert "couldn't find batch" not in result.note.lower()
    # the actual bug this whole plan exists to fix: overwrite must be requested
    assert all(kwargs.get("annotation_overwrite") is True for kwargs in upload_calls)
```

Replace `test_push_version_mixed_new_and_duplicate_uploads_still_files_review_job` (currently lines 1730-1785) with:

```python
def test_push_version_mixed_new_and_duplicate_still_files_review_job_and_reports_both(
    real_db_session, monkeypatch
) -> None:
    """At least one genuinely new upload means the batch really was created
    on Roboflow — the review job must still be filed for it normally, and
    the note must mention both the new upload and the updated duplicate,
    not stay silent just because the review job itself succeeded."""
    import uuid as _uuid

    from app.services.integrations import roboflow_export as mod

    def _fake_write_yolo_dataset(db, *, version_id, root):
        for split in ("train", "valid", "test"):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
        for i in range(2):
            (root / "images" / "train" / f"img{i}.jpg").write_bytes(_jpeg_bytes())
            (root / "labels" / "train" / f"img{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (root / "data.yaml").write_text("names: ['cone']\n", encoding="utf-8")
        return root / "data.yaml"

    monkeypatch.setattr(mod, "write_yolo_dataset", _fake_write_yolo_dataset)

    calls = {"n": 0}

    class _Proj:
        def upload(self, *, batch_name, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return [{"image": {"id": "new-image-id", "success": True, "duplicate": False}}]
            return [{"image": {"id": "existing-image-id", "success": False, "duplicate": True}}]

        def get_batches(self):
            return {"batches": [{"id": "batch-id-1", "name": "autolabelflow", "images": 1}]}

        def create_annotation_job(self, **kwargs):
            return {"id": "job-1"}

    class _WS:
        def project(self, slug):
            return _Proj()

    class _RF:
        def workspace(self, ws):
            return _WS()

    monkeypatch.setattr(mod, "get_client", lambda db: (_RF(), {"default_labeler_email": "a@b.com"}))

    result = mod.push_version_to_roboflow(
        real_db_session,
        version_id=_uuid.uuid4(),
        workspace="ws",
        project_slug="proj",
        custom_batch_name="autolabelflow",
    )

    assert (result.new_images, result.annotations_updated, result.failed) == (1, 1, 0)
    assert result.note is not None
    assert "1 new image" in result.note
    assert "1 existing image" in result.note
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest backend/tests/test_roboflow_jobs.py -k "push_version" -v`
Expected: FAIL — `push_version_to_roboflow` still returns a 4-tuple (`AttributeError: 'tuple' object has no attribute 'new_images'` or similar unpacking error) and never passes `annotation_overwrite`.

- [ ] **Step 3: Implement — add the outcome/result types**

In `backend/app/services/integrations/roboflow_export.py`, add near the top, after the `logger = logging.getLogger(__name__)` line (line 34) and before `_SPLIT_TO_ROBOFLOW`:

```python
class _PushOutcome(Enum):
    """What happened to one pushed image. `ANNOTATION_UPDATED` covers both
    a Roboflow-detected duplicate whose annotation got written via the
    fallback `Project.upload()` path, and (Task 5) an image routed
    directly through `Project.save_annotation()` because we already knew
    its Roboflow id. `UNCHANGED` is a duplicate (or known-id image) with no
    local annotation to push at all — nothing to do, not a failure."""

    NEW_IMAGE = "new_image_uploaded"
    ANNOTATION_UPDATED = "annotation_updated"
    UNCHANGED = "unchanged"


class PushResult(NamedTuple):
    """Return value of `push_version_to_roboflow`. Replaces the old
    `(uploaded, failed, failures, note)` 4-tuple's `uploaded` (which
    conflated genuinely new images with Roboflow-detected duplicates) —
    "duplicate" is expected, not a failure, for this app's own
    import-then-annotate-then-push workflow, so it must never read as
    "nothing happened" when annotations still landed."""

    new_images: int
    annotations_updated: int
    unchanged: int
    failed: int
    failures: list[str]
    note: str | None
```

Add the two new imports at the top of the file (with the existing `import` block, e.g. right after `import time`):

```python
from enum import Enum
from typing import NamedTuple
```

- [ ] **Step 4: Implement — `_upload_one_image` returns `_PushOutcome`, and `_upload_kwargs` always sets `annotation_overwrite=True`**

Replace the current `_upload_one_image` (lines 209-237):

```python
def _upload_one_image(project, **upload_kwargs) -> bool:
    """`project.upload(**upload_kwargs)` with a bounded retry on a transient
    5xx/429 (classified via `status_code`) or a bare connection/timeout
    error (no `status_code` at all — `getattr(exc, "status_code", None)`
    used to fall through to `None`, which isn't in `_UPLOAD_RETRY_STATUSES`,
    so a plain network blip was never retried and immediately failed the
    image). Re-raises the last error once attempts are exhausted, and
    immediately for any non-transient HTTP status. Returns whether Roboflow
    reported this image as a pre-existing duplicate rather than a new
    upload — see `_is_duplicate_upload`."""
    for attempt in range(1, _UPLOAD_MAX_ATTEMPTS + 1):
        try:
            result = project.upload(**upload_kwargs)
            return _is_duplicate_upload(result)
        except Exception as exc:  # noqa: BLE001 - re-raised below, classified by status
            status = getattr(exc, "status_code", None)
            transient = status in _UPLOAD_RETRY_STATUSES or (
                status is None and isinstance(exc, requests.RequestException)
            )
            if not transient or attempt == _UPLOAD_MAX_ATTEMPTS:
                raise
            logger.warning(
                "Roboflow export: %s failed %s (attempt %d/%d) — retrying",
                upload_kwargs.get("image_path"),
                f"HTTP {status}" if status is not None else repr(exc),
                attempt,
                _UPLOAD_MAX_ATTEMPTS,
            )
            time.sleep(_UPLOAD_BACKOFF_BASE_S * 2 ** (attempt - 1))
```

with:

```python
def _upload_one_image(project, *, has_annotation: bool, **upload_kwargs) -> _PushOutcome:
    """`project.upload(**upload_kwargs)` with a bounded retry on a transient
    5xx/429 (classified via `status_code`) or a bare connection/timeout
    error (no `status_code` at all — `getattr(exc, "status_code", None)`
    used to fall through to `None`, which isn't in `_UPLOAD_RETRY_STATUSES`,
    so a plain network blip was never retried and immediately failed the
    image). Re-raises the last error once attempts are exhausted, and
    immediately for any non-transient HTTP status.

    Returns `NEW_IMAGE` for a genuinely new upload, `ANNOTATION_UPDATED`
    when Roboflow reports a pre-existing duplicate AND there was a local
    annotation to push (the SDK's own `single_upload` already resolves the
    duplicate's image id and calls `save_annotation` for it — see
    `_upload_kwargs`'s `annotation_overwrite=True`, without which that
    write silently no-ops on Roboflow's 409), or `UNCHANGED` for a
    duplicate with nothing local to push at all."""
    for attempt in range(1, _UPLOAD_MAX_ATTEMPTS + 1):
        try:
            result = project.upload(**upload_kwargs)
            if _is_duplicate_upload(result):
                return _PushOutcome.ANNOTATION_UPDATED if has_annotation else _PushOutcome.UNCHANGED
            return _PushOutcome.NEW_IMAGE
        except Exception as exc:  # noqa: BLE001 - re-raised below, classified by status
            status = getattr(exc, "status_code", None)
            transient = status in _UPLOAD_RETRY_STATUSES or (
                status is None and isinstance(exc, requests.RequestException)
            )
            if not transient or attempt == _UPLOAD_MAX_ATTEMPTS:
                raise
            logger.warning(
                "Roboflow export: %s failed %s (attempt %d/%d) — retrying",
                upload_kwargs.get("image_path"),
                f"HTTP {status}" if status is not None else repr(exc),
                attempt,
                _UPLOAD_MAX_ATTEMPTS,
            )
            time.sleep(_UPLOAD_BACKOFF_BASE_S * 2 ** (attempt - 1))
```

Now find `_upload_kwargs` (inside `push_version_to_roboflow`, currently lines 332-364) and add `annotation_overwrite=True` to its returned dict — change:

```python
            return dict(
                image_path=str(image_path),
                annotation_path=str(label_path) if has_annotation else None,
                annotation_labelmap=str(data_yaml_path),
                split=roboflow_split,
                batch_name=batch_name,
                # Ground truth is auto-confirmed by Roboflow and skips
                # straight to the "Dataset" column of the Annotate board.
                # These labels come from our pipeline, not a human, so the
                # default target (`ANNOTATING`) instead pushes them as a
                # prediction: Roboflow then queues the image for review
                # rather than treating it as already done. `DATASET` opts
                # into the ground-truth behavior explicitly, when the
                # caller wants to skip that review step.
                is_prediction=upload_target != RoboflowUploadTarget.DATASET.value,
            )
```

to:

```python
            return dict(
                image_path=str(image_path),
                annotation_path=str(label_path) if has_annotation else None,
                annotation_labelmap=str(data_yaml_path),
                split=roboflow_split,
                batch_name=batch_name,
                # Ground truth is auto-confirmed by Roboflow and skips
                # straight to the "Dataset" column of the Annotate board.
                # These labels come from our pipeline, not a human, so the
                # default target (`ANNOTATING`) instead pushes them as a
                # prediction: Roboflow then queues the image for review
                # rather than treating it as already done. `DATASET` opts
                # into the ground-truth behavior explicitly, when the
                # caller wants to skip that review step.
                is_prediction=upload_target != RoboflowUploadTarget.DATASET.value,
                # Without this, Roboflow's 409 "already annotated" response
                # to a duplicate image's annotation write is a silent
                # no-op (see `_upload_one_image`'s docstring) — every
                # re-push of an already-imported, already-annotated image
                # would otherwise never actually update anything on
                # Roboflow. Harmless for a genuinely new image (nothing
                # to overwrite yet).
                annotation_overwrite=True,
            )
```

- [ ] **Step 5: Implement — `_build_export_note` and rewire the main loop's counters**

Add `_build_export_note` near `_assign_annotating_review_job` (e.g. right before it, so it's defined before first use):

```python
def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _build_export_note(
    *, new_images: int, annotations_updated: int, unchanged: int, review_job_note: str | None
) -> str | None:
    """Human-readable summary surfaced as `job.error` (an informational
    note, never a failure — see `run_roboflow_export`'s comment on that
    field). Always non-`None` once at least one image succeeded, so the UI
    shows what actually happened instead of staying silent on a fully
    "clean" push — replaces the old behavior where a successful push with
    no review-job problems reported nothing at all. `review_job_note` is
    whatever `_assign_annotating_review_job` returned for the new-image
    batch (its own failure note, or `None`)."""
    succeeded = new_images + annotations_updated + unchanged
    if succeeded == 0:
        return review_job_note

    parts = []
    if new_images:
        parts.append(f"{_plural(new_images, 'new image')} uploaded")
    if annotations_updated:
        parts.append(f"{_plural(annotations_updated, 'existing image')} updated")
    if unchanged:
        parts.append(f"{_plural(unchanged, 'existing image')} unchanged (no local annotation to push)")
    summary = ", ".join(parts) + "."

    if new_images == 0:
        summary += " No new batch or review job was created — nothing new landed on Roboflow."
    elif review_job_note:
        summary += f" {review_job_note}"
    return summary
```

Now rewire `push_version_to_roboflow`'s body. First, the two early-return guards change shape — replace:

```python
    if should_cancel is not None and should_cancel():
        return 0, 0, [], None
```

with:

```python
    if should_cancel is not None and should_cancel():
        return PushResult(0, 0, 0, 0, [], None)
```

Then the counter setup (currently lines 366-371):

```python
        uploaded = 0
        failed = 0
        duplicates = 0
        failures: list[str] = []
        seen_statuses: list[int | None] = []
        fail_fast_error: RoboflowExportError | None = None
```

becomes:

```python
        new_images = 0
        annotations_updated = 0
        unchanged = 0
        succeeded = 0
        failed = 0
        failures: list[str] = []
        seen_statuses: list[int | None] = []
        fail_fast_error: RoboflowExportError | None = None
```

Then the submit call inside `_top_up` (currently `future = pool.submit(_upload_one_image, project, **_upload_kwargs(split, image_path))`) becomes:

```python
                    kwargs = _upload_kwargs(split, image_path)
                    future = pool.submit(
                        _upload_one_image, project, has_annotation=kwargs["annotation_path"] is not None, **kwargs
                    )
```

Then the per-completion handling (currently):

```python
                    try:
                        is_duplicate = future.result()
                        uploaded += 1
                        if is_duplicate:
                            duplicates += 1
                    except Exception as exc:  # a single bad image shouldn't abort the whole push
```

becomes:

```python
                    try:
                        outcome = future.result()
                        succeeded += 1
                        if outcome is _PushOutcome.NEW_IMAGE:
                            new_images += 1
                        elif outcome is _PushOutcome.ANNOTATION_UPDATED:
                            annotations_updated += 1
                        else:
                            unchanged += 1
                    except Exception as exc:  # a single bad image shouldn't abort the whole push
```

Then `progress_cb(uploaded, total, failed)` (appears twice: the "top-up" line and the fail-fast comment references `uploaded == 0`) — both become `progress_cb(succeeded, total, failed)` and `succeeded == 0` respectively. Find:

```python
                    if progress_cb is not None:
                        progress_cb(uploaded, total, failed)
```

→

```python
                    if progress_cb is not None:
                        progress_cb(succeeded, total, failed)
```

and:

```python
                    if fail_fast_error is None and uploaded == 0 and failed >= _FAIL_FAST_AFTER:
```

→

```python
                    if fail_fast_error is None and succeeded == 0 and failed >= _FAIL_FAST_AFTER:
```

Finally, replace the whole tail block (currently lines 439-463):

```python
    logger.info(
        "Roboflow export finished: %d uploaded, %d failed (batch=%r)", uploaded, failed, batch_name
    )

    annotation_job_note: str | None = None
    new_uploads = uploaded - duplicates
    if uploaded > 0 and new_uploads == 0:
        # Every "successful" upload was actually Roboflow reporting the
        # image's content already exists in this project — no batch was
        # ever created under `batch_name` for this push, so searching for
        # one (`_assign_annotating_review_job`) would only produce the
        # misleading "couldn't find batch ... sitting in Unassigned"
        # message when in fact nothing new landed anywhere.
        annotation_job_note = (
            f"All {duplicates} image(s) were skipped as duplicates already present in this "
            "Roboflow project (Roboflow doesn't re-add an image whose file content exactly "
            "matches one already there) — no new images were added, so no batch or review job "
            "was created."
        )
    elif upload_target == RoboflowUploadTarget.ANNOTATING.value and uploaded > 0:
        annotation_job_note = _assign_annotating_review_job(
            project, batch_name=batch_name, labeler_email=config.get("default_labeler_email")
        )

    return uploaded, failed, failures, annotation_job_note
```

with:

```python
    logger.info(
        "Roboflow export finished: %d new, %d annotations updated, %d unchanged, %d failed (batch=%r)",
        new_images,
        annotations_updated,
        unchanged,
        failed,
        batch_name,
    )

    review_job_note: str | None = None
    if upload_target == RoboflowUploadTarget.ANNOTATING.value and new_images > 0:
        # Only ever called when something new actually needs moving out of
        # Unassigned — a duplicate/known-id annotation update never joins
        # the batch this looks up, so there'd be nothing to find/move for
        # an all-existing-images push (see `_build_export_note` for how
        # that case is reported instead).
        review_job_note = _assign_annotating_review_job(
            project, batch_name=batch_name, labeler_email=config.get("default_labeler_email")
        )

    note = _build_export_note(
        new_images=new_images,
        annotations_updated=annotations_updated,
        unchanged=unchanged,
        review_job_note=review_job_note,
    )

    return PushResult(new_images, annotations_updated, unchanged, failed, failures, note)
```

- [ ] **Step 6: Update `push_version_to_roboflow`'s docstring**

Its docstring (lines 251-277) still describes the old `uploaded`/`duplicates` model in its opening line and in the `progress_cb` paragraph. Update the first line:

```python
    """Returns (uploaded_count, failed_count, failure_messages, annotation_job_note).
```

to:

```python
    """Returns a `PushResult` — new-image / annotation-updated / unchanged /
    failed counts, the failure messages, and a human-readable summary note.
```

and in the `progress_cb` paragraph, `` `uploaded` counts only images that actually reached Roboflow `` stays accurate (rename `uploaded` → `succeeded` in that sentence for consistency with the new local variable name, purely cosmetic).

- [ ] **Step 7: Update `run_roboflow_export` for the new return type**

In `backend/app/workers/tasks/roboflow.py`, replace (lines 141-153):

```python
        uploaded, failed, failures, annotation_job_note = push_version_to_roboflow(
            db,
            version_id=job.dataset_version_id,
            workspace=job.workspace,
            project_slug=job.project_slug,
            custom_batch_name=job.batch_name,
            upload_target=job.upload_target,
            progress_cb=_make_progress_cb(job, db, writer),
            should_cancel=lambda: is_cancel_requested(job_id),
        )
        job.uploaded_count = uploaded
        job.failed_count = failed
        job.failures = failures
```

with:

```python
        result = push_version_to_roboflow(
            db,
            version_id=job.dataset_version_id,
            workspace=job.workspace,
            project_slug=job.project_slug,
            custom_batch_name=job.batch_name,
            upload_target=job.upload_target,
            progress_cb=_make_progress_cb(job, db, writer),
            should_cancel=lambda: is_cancel_requested(job_id),
        )
        job.new_images_count = result.new_images
        job.annotations_updated_count = result.annotations_updated
        job.uploaded_count = result.new_images + result.annotations_updated + result.unchanged
        job.failed_count = result.failed
        job.failures = result.failures
```

Then further down, replace every remaining bare `uploaded`/`failed`/`failures`/`annotation_job_note` reference in this same function (lines ~154-181) with `job.uploaded_count`/`result.failed`/`result.failures`/`result.note` respectively:

```python
        if is_cancel_requested(job_id):
            job.status = RoboflowJobStatus.CANCELLED
            db.commit()
            writer.finish(status="CANCELLED")
        elif uploaded == 0 and failed > 0:
            # Every image failed, but the version was smaller than the
            # service's fail-fast threshold so it returned instead of
            # raising. Still a failed export, not a COMPLETED one that
            # silently pushed nothing.
            job.status = RoboflowJobStatus.FAILED
            job.error = (
                f"All {failed} image(s) failed to upload to Roboflow — nothing was pushed. "
                f"First error: {failures[0]}"
                if failures
                else f"All {failed} image(s) failed to upload to Roboflow — nothing was pushed."
            )[:_MAX_ERROR_LEN]
            db.commit()
            writer.finish(status="FAILED", error=job.error)
        else:
            job.status = RoboflowJobStatus.COMPLETED
            # Informational only (e.g. "uploaded fine, but couldn't
            # auto-create the Annotating review job") — never flips status
            # away from COMPLETED, the images themselves are already on
            # Roboflow either way.
            if annotation_job_note:
                job.error = annotation_job_note[:_MAX_ERROR_LEN]
            db.commit()
            writer.finish(error=annotation_job_note)
```

becomes:

```python
        if is_cancel_requested(job_id):
            job.status = RoboflowJobStatus.CANCELLED
            db.commit()
            writer.finish(status="CANCELLED")
        elif job.uploaded_count == 0 and result.failed > 0:
            # Every image failed, but the version was smaller than the
            # service's fail-fast threshold so it returned instead of
            # raising. Still a failed export, not a COMPLETED one that
            # silently pushed nothing.
            job.status = RoboflowJobStatus.FAILED
            job.error = (
                f"All {result.failed} image(s) failed to upload to Roboflow — nothing was pushed. "
                f"First error: {result.failures[0]}"
                if result.failures
                else f"All {result.failed} image(s) failed to upload to Roboflow — nothing was pushed."
            )[:_MAX_ERROR_LEN]
            db.commit()
            writer.finish(status="FAILED", error=job.error)
        else:
            job.status = RoboflowJobStatus.COMPLETED
            # Informational only (e.g. "3 new images uploaded, 7 existing
            # images updated" or "couldn't auto-create the Annotating
            # review job") — never flips status away from COMPLETED, the
            # images themselves are already on Roboflow either way.
            if result.note:
                job.error = result.note[:_MAX_ERROR_LEN]
            db.commit()
            writer.finish(error=result.note)
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest backend/tests/test_roboflow_jobs.py -v`
Expected: all PASS, including the two rewritten `test_push_version_*` tests and every pre-existing `test_roboflow_export_*` test (they only ever asserted `uploaded_count`/`failed_count` via the HTTP API, which still work identically since `job.uploaded_count` keeps the same total-success meaning).

- [ ] **Step 9: Commit**

```bash
git add backend/app/services/integrations/roboflow_export.py backend/app/workers/tasks/roboflow.py backend/tests/test_roboflow_jobs.py
git commit -m "fix(roboflow): stop silently dropping annotation updates on re-pushed duplicates

Roboflow's save_annotation defaults to overwrite=False, so re-pushing an
already-annotated image (the normal import-then-annotate-then-push flow)
hit a 409 'already annotated' that the SDK swallowed into a no-op. Also
replaces the uploaded/duplicates counters with honest
new_image/annotation_updated/unchanged/failed outcome tracking and a
summary note that's never silently empty on a successful push."
```

---

### Task 5: Direct routing — skip `Project.upload()` for known Roboflow images

**Files:**
- Modify: `backend/app/services/integrations/roboflow_export.py`
- Test: `backend/tests/test_roboflow_jobs.py`, `backend/tests/test_roboflow_jobs.py` (extend `_FakeRoboflowProject`)

**Interfaces:**
- Consumes: `Image.roboflow_image_id`/`roboflow_workspace`/`roboflow_project_slug` (Task 1, populated by Task 3). `_PushOutcome`, `_upload_one_image`, `_UPLOAD_RETRY_STATUSES`, `_UPLOAD_MAX_ATTEMPTS`, `_UPLOAD_BACKOFF_BASE_S` (Task 4).
- Produces: `_save_annotation_only(project, *, roboflow_image_id: str, annotation_path: str, annotation_labelmap_path: str, batch_name: str, is_prediction: bool) -> _PushOutcome`, `_push_image(project, *, roboflow_image_id: str | None, upload_kwargs: dict, batch_name: str, is_prediction: bool, annotation_labelmap_path: str) -> _PushOutcome` — the new per-image dispatch entry point the main loop submits to the thread pool instead of `_upload_one_image` directly.

- [ ] **Step 1: Extend the test double with `save_annotation`**

In `backend/tests/test_roboflow_jobs.py`, add a `save_annotation` method to `_FakeRoboflowProject` (after its existing `upload` method, around line 182):

```python
        self.saved_annotations: list[dict] = []

    def save_annotation(
        self,
        *,
        annotation_path: str,
        annotation_labelmap,
        image_id: str,
        job_name: str | None = None,
        is_prediction: bool = False,
        annotation_overwrite: bool = False,
    ) -> dict:
        if image_id == "missing-on-roboflow":
            from roboflow.adapters.rfapi import AnnotationSaveError

            raise AnnotationSaveError("not found", status_code=404)
        self.saved_annotations.append(
            {
                "annotation_path": annotation_path,
                "annotation_labelmap": annotation_labelmap,
                "image_id": image_id,
                "job_name": job_name,
                "is_prediction": is_prediction,
                "annotation_overwrite": annotation_overwrite,
            }
        )
        return {"success": True}
```

(Add `self.saved_annotations: list[dict] = []` in `__init__` right after `self.annotation_jobs: list[dict] = []`, not inline where shown above — put the method body's `save_annotation` def directly after `__init__`/`upload`, and the list initialization inside `__init__`.)

- [ ] **Step 2: Write the failing tests**

Add to `backend/tests/test_roboflow_jobs.py` (these use `connected_roboflow`/real HTTP + Celery-eager, same as the other `test_roboflow_export_*` tests, so the DB `Image` row and the materialized export file's name naturally match via the real `write_yolo_dataset`/`out_image_filename` convention — no fake `write_yolo_dataset` needed here):

```python
def test_roboflow_export_known_image_id_updates_annotation_without_reupload(
    connected_roboflow: TestClient, unique_name: str
) -> None:
    """An image this app already knows the Roboflow id for (imported from
    the same workspace/project being exported to) must be routed straight
    through save_annotation — Project.upload() must never be called for
    it at all. Builds the exportable version through the exact same real
    API calls as the `approved_version` fixture above (`real_client.post`
    for project/dataset/image/approve/annotation/version), then sets
    `roboflow_image_id`/`roboflow_workspace`/`roboflow_project_slug` on
    the `Image` row directly afterward — there is no API for setting those
    outside the real Roboflow import flow, which this test deliberately
    isn't invoking (it only needs the *result* of having imported, not the
    import itself — Task 3 already covers the import path writing these)."""
    from app.db.session import SessionLocal
    from app.models.image import Image

    project = connected_roboflow.post("/api/v1/projects", json={"name": unique_name}).json()
    connected_roboflow.patch(
        f"/api/v1/projects/{project['id']}", json={"class_config": [{"id": 0, "name": "cone"}]}
    )
    dataset = connected_roboflow.post(f"/api/v1/projects/{project['id']}/datasets", json={"name": "d"}).json()
    image = connected_roboflow.post(
        f"/api/v1/datasets/{dataset['id']}/images", files={"file": ("f.jpg", _jpeg_bytes(), "image/jpeg")}
    ).json()
    connected_roboflow.post(f"/api/v1/images/{image['id']}/approve")
    connected_roboflow.post(
        "/api/v1/annotations",
        json={
            "image_id": image["id"],
            "class_id": 0,
            "class_name": "cone",
            "x1": 0.1,
            "y1": 0.1,
            "x2": 0.3,
            "y2": 0.3,
        },
    )

    db = SessionLocal()
    try:
        db.query(Image).filter(Image.id == image["id"]).update(
            {
                "roboflow_image_id": "known-rf-image-id",
                "roboflow_workspace": "my-workspace",
                "roboflow_project_slug": "cones",
            }
        )
        db.commit()
    finally:
        db.close()

    version = connected_roboflow.post(f"/api/v1/datasets/{dataset['id']}/versions", json={}).json()

    resp = connected_roboflow.post(
        f"/api/v1/versions/{version['id']}/export/roboflow",
        json={"workspace": "my-workspace", "project": "cones"},
    )
    assert resp.status_code == 202, resp.text
    job = resp.json()
    assert job["status"] == "COMPLETED", job
    assert job["new_images_count"] == 0
    assert job["annotations_updated_count"] == 1
```

Also add the lower-level unit test (mirrors the existing `test_push_version_*` style, faster and more precise about what's being routed):

```python
def test_push_version_known_roboflow_id_routes_through_save_annotation_not_upload(
    real_db_session, monkeypatch, unique_name: str
) -> None:
    """Core of Task 5: an Image row with a roboflow_image_id matching the
    export target's workspace/project must skip Project.upload() entirely
    and call Project.save_annotation() directly instead."""
    import uuid as _uuid

    from app.models.dataset import Dataset
    from app.models.image import Image, ImageSourceType
    from app.models.project import Project
    from app.services.integrations import roboflow_export as mod

    project = Project(name=unique_name, slug=unique_name, class_config=[{"id": 0, "name": "cone"}])
    real_db_session.add(project)
    real_db_session.flush()
    dataset = Dataset(project_id=project.id, name="ds")
    real_db_session.add(dataset)
    real_db_session.flush()
    image = Image(
        project_id=project.id,
        dataset_id=dataset.id,
        storage_key="k",
        original_filename="a.jpg",
        width=64,
        height=48,
        source_type=ImageSourceType.UPLOAD,
        roboflow_image_id="known-rf-image-id",
        roboflow_workspace="ws",
        roboflow_project_slug="proj",
    )
    real_db_session.add(image)
    real_db_session.commit()
    real_db_session.refresh(image)

    def _fake_write_yolo_dataset(db, *, version_id, root):
        (root / "images" / "train").mkdir(parents=True)
        (root / "labels" / "train").mkdir(parents=True)
        (root / "images" / "valid").mkdir(parents=True)
        (root / "labels" / "valid").mkdir(parents=True)
        (root / "images" / "test").mkdir(parents=True)
        (root / "labels" / "test").mkdir(parents=True)
        (root / "images" / "train" / f"{image.id}.jpg").write_bytes(_jpeg_bytes())
        (root / "labels" / "train" / f"{image.id}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (root / "data.yaml").write_text("names: ['cone']\n", encoding="utf-8")
        return root / "data.yaml"

    monkeypatch.setattr(mod, "write_yolo_dataset", _fake_write_yolo_dataset)

    upload_calls = []
    save_calls = []

    class _Proj:
        def upload(self, **kwargs):
            upload_calls.append(kwargs)
            return [{"image": {"id": "should-not-be-called", "success": True, "duplicate": False}}]

        def save_annotation(self, **kwargs):
            save_calls.append(kwargs)
            return {"success": True}

        def get_batches(self):
            return {"batches": []}

    class _WS:
        def project(self, slug):
            return _Proj()

    class _RF:
        def workspace(self, ws):
            return _WS()

    monkeypatch.setattr(mod, "get_client", lambda db: (_RF(), {"default_labeler_email": "a@b.com"}))

    result = mod.push_version_to_roboflow(
        real_db_session, version_id=_uuid.uuid4(), workspace="ws", project_slug="proj"
    )

    assert not upload_calls
    assert len(save_calls) == 1
    assert save_calls[0]["image_id"] == "known-rf-image-id"
    assert save_calls[0]["annotation_overwrite"] is True
    assert save_calls[0]["annotation_labelmap"] == {0: "cone"}  # loaded, not the bare yaml path
    assert (result.new_images, result.annotations_updated, result.unchanged, result.failed) == (0, 1, 0, 0)


def test_push_version_known_roboflow_id_different_project_falls_back_to_upload(
    real_db_session, monkeypatch, unique_name: str
) -> None:
    """A stored id from a *different* Roboflow project than the one being
    exported to must not be trusted — falls back to a normal upload."""
    import uuid as _uuid

    from app.models.dataset import Dataset
    from app.models.image import Image, ImageSourceType
    from app.models.project import Project
    from app.services.integrations import roboflow_export as mod

    project = Project(name=unique_name, slug=unique_name, class_config=[{"id": 0, "name": "cone"}])
    real_db_session.add(project)
    real_db_session.flush()
    dataset = Dataset(project_id=project.id, name="ds")
    real_db_session.add(dataset)
    real_db_session.flush()
    image = Image(
        project_id=project.id,
        dataset_id=dataset.id,
        storage_key="k",
        original_filename="a.jpg",
        width=64,
        height=48,
        source_type=ImageSourceType.UPLOAD,
        roboflow_image_id="known-rf-image-id",
        roboflow_workspace="ws",
        roboflow_project_slug="a-different-project",  # <- mismatch
    )
    real_db_session.add(image)
    real_db_session.commit()
    real_db_session.refresh(image)

    def _fake_write_yolo_dataset(db, *, version_id, root):
        (root / "images" / "train").mkdir(parents=True)
        (root / "labels" / "train").mkdir(parents=True)
        (root / "images" / "valid").mkdir(parents=True)
        (root / "labels" / "valid").mkdir(parents=True)
        (root / "images" / "test").mkdir(parents=True)
        (root / "labels" / "test").mkdir(parents=True)
        (root / "images" / "train" / f"{image.id}.jpg").write_bytes(_jpeg_bytes())
        (root / "labels" / "train" / f"{image.id}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (root / "data.yaml").write_text("names: ['cone']\n", encoding="utf-8")
        return root / "data.yaml"

    monkeypatch.setattr(mod, "write_yolo_dataset", _fake_write_yolo_dataset)

    upload_calls = []
    save_calls = []

    class _Proj:
        def upload(self, **kwargs):
            upload_calls.append(kwargs)
            return [{"image": {"id": "new-id", "success": True, "duplicate": False}}]

        def save_annotation(self, **kwargs):
            save_calls.append(kwargs)
            return {"success": True}

        def get_batches(self):
            return {"batches": []}

    class _WS:
        def project(self, slug):
            return _Proj()

    class _RF:
        def workspace(self, ws):
            return _WS()

    monkeypatch.setattr(mod, "get_client", lambda db: (_RF(), {"default_labeler_email": "a@b.com"}))

    result = mod.push_version_to_roboflow(
        real_db_session, version_id=_uuid.uuid4(), workspace="ws", project_slug="proj"
    )

    assert not save_calls
    assert len(upload_calls) == 1
    assert result.new_images == 1


def test_push_version_stale_known_roboflow_id_404_falls_back_to_upload(
    real_db_session, monkeypatch, unique_name: str
) -> None:
    """The stored id no longer resolves on Roboflow (image deleted there
    since import) — save_annotation 404s, and that one image falls back to
    a normal upload instead of failing the whole export."""
    import uuid as _uuid

    from app.models.dataset import Dataset
    from app.models.image import Image, ImageSourceType
    from app.models.project import Project
    from app.services.integrations import roboflow_export as mod

    project = Project(name=unique_name, slug=unique_name, class_config=[{"id": 0, "name": "cone"}])
    real_db_session.add(project)
    real_db_session.flush()
    dataset = Dataset(project_id=project.id, name="ds")
    real_db_session.add(dataset)
    real_db_session.flush()
    image = Image(
        project_id=project.id,
        dataset_id=dataset.id,
        storage_key="k",
        original_filename="a.jpg",
        width=64,
        height=48,
        source_type=ImageSourceType.UPLOAD,
        roboflow_image_id="missing-on-roboflow",
        roboflow_workspace="ws",
        roboflow_project_slug="proj",
    )
    real_db_session.add(image)
    real_db_session.commit()
    real_db_session.refresh(image)

    def _fake_write_yolo_dataset(db, *, version_id, root):
        (root / "images" / "train").mkdir(parents=True)
        (root / "labels" / "train").mkdir(parents=True)
        (root / "images" / "valid").mkdir(parents=True)
        (root / "labels" / "valid").mkdir(parents=True)
        (root / "images" / "test").mkdir(parents=True)
        (root / "labels" / "test").mkdir(parents=True)
        (root / "images" / "train" / f"{image.id}.jpg").write_bytes(_jpeg_bytes())
        (root / "labels" / "train" / f"{image.id}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (root / "data.yaml").write_text("names: ['cone']\n", encoding="utf-8")
        return root / "data.yaml"

    monkeypatch.setattr(mod, "write_yolo_dataset", _fake_write_yolo_dataset)

    upload_calls = []

    class _Proj:
        def upload(self, **kwargs):
            upload_calls.append(kwargs)
            return [{"image": {"id": "new-id", "success": True, "duplicate": False}}]

        def save_annotation(self, **kwargs):
            from roboflow.adapters.rfapi import AnnotationSaveError

            raise AnnotationSaveError("not found", status_code=404)

        def get_batches(self):
            return {"batches": []}

    class _WS:
        def project(self, slug):
            return _Proj()

    class _RF:
        def workspace(self, ws):
            return _WS()

    monkeypatch.setattr(mod, "get_client", lambda db: (_RF(), {"default_labeler_email": "a@b.com"}))

    result = mod.push_version_to_roboflow(
        real_db_session, version_id=_uuid.uuid4(), workspace="ws", project_slug="proj"
    )

    assert len(upload_calls) == 1
    assert result.new_images == 1
    assert result.failed == 0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest backend/tests/test_roboflow_jobs.py -k "known_roboflow_id or known_image_id" -v`
Expected: FAIL — nothing yet reads `Image.roboflow_image_id` inside `push_version_to_roboflow`, so `_Proj.upload` gets called for every image and `save_calls`/`upload_calls` assertions fail.

- [ ] **Step 4: Implement `_save_annotation_only`**

In `backend/app/services/integrations/roboflow_export.py`, add right after `_upload_one_image`:

```python
def _save_annotation_only(
    project,
    *,
    roboflow_image_id: str,
    annotation_path: str,
    annotation_labelmap_path: str,
    batch_name: str,
    is_prediction: bool,
) -> _PushOutcome:
    """Updates the annotation on an already-known Roboflow image directly,
    via `Project.save_annotation()` — never `Project.upload()` — so the
    image's (unchanged) bytes are never re-sent and Roboflow's
    duplicate-detection is never involved at all. `annotation_overwrite`
    is always `True`: the whole point of this path is replacing whatever
    annotation the image already carries (the ground truth pulled in at
    import time, or a prediction from an earlier push) with the current
    local one — the default `overwrite=False` would instead hit Roboflow's
    409 "already annotated" response and silently do nothing (see
    `rfapi.save_annotation`, `roboflow/adapters/rfapi.py:815-821`).

    `annotation_labelmap` must be the loaded `{index: name}` mapping, not
    a bare yaml path — `Project.upload()`'s own internal call chain
    (`single_upload`, `roboflow/core/project.py:604-605`) converts it via
    `load_labelmap()` before ever calling `save_annotation`; calling
    `Project.save_annotation()` directly, as this does, means doing that
    conversion ourselves or Roboflow receives the literal path string as
    the labelmap and silently mislabels every class.

    Retries a transient 5xx/429 or bare network error the same as
    `_upload_one_image`; anything else (in particular a 404 — the stored
    id no longer resolves to an image on Roboflow) is raised for the
    caller (`_push_image`) to fall back to a normal upload for."""
    from roboflow.util.image_utils import load_labelmap

    labelmap = load_labelmap(annotation_labelmap_path)
    for attempt in range(1, _UPLOAD_MAX_ATTEMPTS + 1):
        try:
            project.save_annotation(
                annotation_path=annotation_path,
                annotation_labelmap=labelmap,
                image_id=roboflow_image_id,
                job_name=batch_name,
                is_prediction=is_prediction,
                annotation_overwrite=True,
            )
            return _PushOutcome.ANNOTATION_UPDATED
        except Exception as exc:  # noqa: BLE001 - re-raised below, classified by status
            status = getattr(exc, "status_code", None)
            transient = status in _UPLOAD_RETRY_STATUSES or (
                status is None and isinstance(exc, requests.RequestException)
            )
            if not transient or attempt == _UPLOAD_MAX_ATTEMPTS:
                raise
            logger.warning(
                "Roboflow export: annotation update for Roboflow image %s failed %s "
                "(attempt %d/%d) — retrying",
                roboflow_image_id,
                f"HTTP {status}" if status is not None else repr(exc),
                attempt,
                _UPLOAD_MAX_ATTEMPTS,
            )
            time.sleep(_UPLOAD_BACKOFF_BASE_S * 2 ** (attempt - 1))
```

- [ ] **Step 5: Implement `_push_image` (the dispatcher)**

Add right after `_save_annotation_only`:

```python
def _push_image(
    project,
    *,
    roboflow_image_id: str | None,
    upload_kwargs: dict,
    batch_name: str,
    is_prediction: bool,
    annotation_labelmap_path: str,
) -> _PushOutcome:
    """Routes a single image to whichever Roboflow write path applies. An
    image this app already knows the Roboflow id for (imported from this
    same workspace/project, via `import_roboflow_raw_project` — see
    `Image.roboflow_image_id`) skips `Project.upload()` entirely and only
    ever updates its annotation through `_save_annotation_only` — never
    re-uploading bytes Roboflow already has, never depending on its
    duplicate-detection. Everything else (a genuinely new local image, an
    image with no local annotation to push at all, or a known id that
    404s because the Roboflow image was deleted since import) falls
    through to the normal `_upload_one_image` path."""
    has_annotation = upload_kwargs["annotation_path"] is not None
    if roboflow_image_id is not None:
        if not has_annotation:
            return _PushOutcome.UNCHANGED
        try:
            return _save_annotation_only(
                project,
                roboflow_image_id=roboflow_image_id,
                annotation_path=upload_kwargs["annotation_path"],
                annotation_labelmap_path=annotation_labelmap_path,
                batch_name=batch_name,
                is_prediction=is_prediction,
            )
        except Exception as exc:  # noqa: BLE001 - only a 404 falls through; anything else is a real failure
            if getattr(exc, "status_code", None) != 404:
                raise
            logger.warning(
                "Roboflow export: stored Roboflow image id %s no longer exists on Roboflow "
                "(404) — falling back to a fresh upload for %s",
                roboflow_image_id,
                upload_kwargs.get("image_path"),
            )
    return _upload_one_image(project, has_annotation=has_annotation, **upload_kwargs)
```

- [ ] **Step 6: Wire the DB lookup and route the main loop through `_push_image`**

Add the imports at the top of `roboflow_export.py` (with the existing `app.models.*` imports):

```python
from app.models.image import Image
```

and (with the existing `sqlalchemy.orm` import):

```python
from sqlalchemy import select
```

Inside `push_version_to_roboflow`, right after `pending` is fully built and `total`/the first `progress_cb(0, total, 0)` call (i.e., right after what's currently lines 328-330, before `def _upload_kwargs`), add the bulk lookup:

```python
        def _image_uuid(image_path: Path) -> uuid.UUID | None:
            # Every materialized export file is named after its own
            # `Image.id` (see `version_data.out_image_filename` —
            # `f"{image.id}{ext}"`), so the stem is always a real UUID in
            # production. A test double that fakes `write_yolo_dataset`
            # with synthetic names (e.g. "img0.jpg") is the one case this
            # legitimately fails to parse — treated the same as "no known
            # Roboflow id for this image", which is the correct fallback.
            try:
                return uuid.UUID(image_path.stem)
            except ValueError:
                return None

        pending_uuids = {u for u in (_image_uuid(p) for _split, p in pending) if u is not None}
        known_roboflow_ids: dict[uuid.UUID, str] = {}
        if pending_uuids:
            rows = db.execute(
                select(Image.id, Image.roboflow_image_id).where(
                    Image.id.in_(pending_uuids),
                    Image.roboflow_workspace == workspace,
                    Image.roboflow_project_slug == project_slug,
                    Image.roboflow_image_id.is_not(None),
                )
            ).all()
            known_roboflow_ids = {row.id: row.roboflow_image_id for row in rows}
```

Then change the `_top_up` submit call (from Task 4's Step 5) from:

```python
                    kwargs = _upload_kwargs(split, image_path)
                    future = pool.submit(
                        _upload_one_image, project, has_annotation=kwargs["annotation_path"] is not None, **kwargs
                    )
```

to:

```python
                    kwargs = _upload_kwargs(split, image_path)
                    future = pool.submit(
                        _push_image,
                        project,
                        roboflow_image_id=known_roboflow_ids.get(_image_uuid(image_path)),
                        upload_kwargs=kwargs,
                        batch_name=batch_name,
                        is_prediction=kwargs["is_prediction"],
                        annotation_labelmap_path=kwargs["annotation_labelmap"],
                    )
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest backend/tests/test_roboflow_jobs.py -v`
Expected: all PASS. If `test_roboflow_export_known_image_id_updates_annotation_without_reupload` (the HTTP-level test from Step 2) still fails on fixture-shape mismatches, fix its fixture calls to match this repo's actual dataset-version/approval endpoints (see that test's trailing note) rather than skipping it — the three lower-level `test_push_version_known_roboflow_id_*` tests are the ones that must pass without any fixture guessing, since they drive `push_version_to_roboflow` directly.

- [ ] **Step 8: Commit**

```bash
git add backend/app/services/integrations/roboflow_export.py backend/tests/test_roboflow_jobs.py
git commit -m "feat(roboflow): route already-imported images through save_annotation directly

Images we already know the Roboflow id for (imported from the same
workspace/project via import_roboflow_raw_project) now skip
Project.upload() entirely on re-push — no re-uploading unchanged bytes,
no dependency on Roboflow's duplicate detection. Falls back to a normal
upload on a workspace/project mismatch or a stale (404) id."
```

---

### Task 6: Frontend — surface the new counts on `RoboflowJob`

**Files:**
- Modify: `frontend/src/types/index.ts`

**Interfaces:**
- Consumes: `RoboflowJobRead`'s new `new_images_count`/`annotations_updated_count` fields (Task 2).

- [ ] **Step 1: Update the type**

In `frontend/src/types/index.ts`, around line 343 (`uploaded_count: number;`), add the two new fields:

```typescript
  uploaded_count: number;
  new_images_count: number;
  annotations_updated_count: number;
  failed_count: number;
```

No component changes are required for correctness: `RoboflowJobProgress.tsx` already renders `job.error`/the streamed `progress.error` verbatim (`RoboflowJobProgress.tsx:124`), and Task 4/5's `_build_export_note` now always populates that field with the honest "N new images uploaded, N existing images updated" breakdown — the existing generic rendering path picks it up with no code change. Adding the typed fields here is for any future UI work (e.g. separate stat chips) to build on without re-deriving them from the note string.

- [ ] **Step 2: Verify the frontend typechecks**

Run: `cd frontend && npm run typecheck` (or the project's equivalent — check `AGENTS.md` for the exact command if this one doesn't exist)
Expected: no new type errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/types/index.ts
git commit -m "feat(roboflow): expose new_images_count/annotations_updated_count on RoboflowJob type"
```

---

## Manual Verification (after all tasks)

1. Run the full backend suite once: `cd backend && pytest -v` — confirm no regressions outside `test_roboflow_jobs.py` (nothing else imports `push_version_to_roboflow` or `Image` in a way this plan's changes could break, but this is the cheap way to be sure).
2. Re-run the exact scenario from the bug report: import a Roboflow project (raw pull, no Version), annotate an image in the app, export back to the same workspace/project. Confirm:
   - The job's `annotations_updated_count` is 1, `new_images_count` is 0.
   - The job's note reads something like "1 existing image updated. No new batch or review job was created — nothing new landed on Roboflow." — not "skipped as duplicates."
   - In Roboflow itself, the image's annotation actually changed (this is the one step no automated test in this plan can verify end-to-end against the real Roboflow API — use `backend/scripts/roboflow_check_annotations.py` against the real workspace, or check the image directly in Roboflow's UI).
