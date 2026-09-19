# Move Annotation-Updated Images to Annotating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a Roboflow export updates the annotation on images this app already knew about (the "known Roboflow id" path from the prior `fix/roboflow-annotation-only-repush` branch), move exactly those images out of Roboflow's Unassigned column into a named Annotating review job — the same way genuinely new images already do — without disturbing any other image in the project.

**Architecture:** Roboflow has no "create a job for these arbitrary existing images" call. The only path is: find which batch an image currently belongs to, carve the subset we care about out of that batch into a new named batch (`Project.create_annotation_batch`), merge multiple such carved batches together if more than one source batch is involved (`Project.merge_annotation_batches`), then file a review job against the resulting batch (`Project.create_annotation_job` — already used for new images). Finding "which batch is image X in" has no direct API either, so it's discovered by paginating each of the project's batches via Roboflow's `/search` endpoint and checking each page's ids against the set we're looking for, stopping as soon as every target id has been found. All of this was verified against the real, installed SDK and a live test run against a real Roboflow project during this plan's design — see the Global Constraints below for the confirmed request/response shapes.

**Tech Stack:** FastAPI + SQLAlchemy (backend), the installed `roboflow` Python SDK (`backend/venv/Lib/site-packages/roboflow`), pytest.

**Spec:** This plan is written directly against the live-verified Roboflow API and the existing `backend/app/services/integrations/roboflow_export.py` (the `fix/roboflow-annotation-only-repush` branch, already merged into `main`) — no separate spec doc exists. User-confirmed decisions this plan encodes: (1) moving updated images to Annotating happens automatically on every export where `upload_target=ANNOTATING` and a labeler email is configured — same trigger as new images already use, no opt-in flag; (2) the carved batch/job for updated images reuses the same `batch_name` already computed for the export (no separate "-updated" suffix).

## Global Constraints

- **Live-verified Roboflow API shapes** (confirmed against a real project, `gtps-workspace/fitfiscone_ball`, during this plan's design — do not deviate from these without re-verifying against the installed SDK):
  - `Project.create_annotation_batch(source_batch_id: str, image_ids: list[str], name: str | None) -> dict` returns `{"batchId": "<new batch id>", "movedImageCount": <int>}` on success. Confirmed live: carving 2 images out of an 11-image batch left 9 in the source and created a new batch with those 2.
  - `Project.merge_annotation_batches(source_batch_ids: list[str], target_batch_id: str) -> dict` — not live-tested in this plan's design pass (only single-batch carve was tested), but its SDK signature/implementation (`roboflow/core/project.py`, `roboflow/adapters/rfapi.py`) was read and matches `create_annotation_batch`'s calling convention exactly.
  - `Project.create_annotation_job(batch_id, labeler_email, reviewer_email, name=None) -> dict` (already used in this codebase) — confirmed live: job created from a carved batch shows `"numImages": 2, "unannotated": 0, "annotated": 2, "status": "assigned"`.
  - Roboflow's `/search` endpoint has **no field that reports an image's batch membership** and **no "look up these specific ids" filter** — confirmed live (`Invalid fields requested: batch, annotation. Valid fields are: annotations, aspectRatio, created, embedding, features, filename, height, id, labels, name, owner, projects, score, split, tags, url, user_metadata, width, offset, limit`) and confirmed against a real image's full detail payload (`GET .../images/<id>`) which also carries no batch field. The only way to find an image's batch is to page through a specific batch's own image list (`/search` with `batch: true, batch_id: <id>`) and check.
- Backend tests run via `cd backend && DATABASE_URL=postgresql+psycopg://annotate:annotate@localhost:5432/annotate_test ./venv/Scripts/python.exe -m pytest tests/ -q` (or without `DATABASE_URL` for the SQLite desktop-profile default — this repo's test env in practice always has `DATABASE_URL` pointed at a real Postgres already; FK constraints are enforced either way).
- **Scope boundary, stated explicitly so no task accidentally widens it:** only images pushed through the known-Roboflow-id direct path (`_save_annotation_only`, i.e. `Image.roboflow_image_id` was set and matched the export's target workspace/project) feed this new carve-to-job step. Images whose annotation got updated via the SDK's own duplicate-detection fallback (`_upload_one_image`'s duplicate branch, used when there's no stored id) are **not** included — discovering their Roboflow id would need extra plumbing (the id only appears in that call's own response, never surfaced today) for what is already the rarer case. They keep today's behavior: annotation updated, image stays wherever it already was.
- Existing tests that must keep passing unmodified (verified during this plan's design by tracing exactly which code path each exercises): `test_roboflow_export_annotating_without_labeler_email_notes_it_non_fatally`, `test_roboflow_export_annotating_creates_review_job_when_labeler_email_set`, `test_roboflow_export_dataset_target_never_creates_annotation_job`, `test_push_version_all_duplicates_updates_annotations_not_skipped`, `test_push_version_mixed_new_and_duplicate_still_files_review_job_and_reports_both` — none of these exercise the known-Roboflow-id path, so `updated_known_roboflow_ids` stays empty for all of them and the new code path is never entered. `test_push_version_known_roboflow_id_routes_through_save_annotation_not_upload` **does** exercise the known-id path and its fake `_Proj.get_batches()` already returns `{"batches": []}` — after this plan's change it will additionally be asked to satisfy a batch-membership discovery call; verify in Task 3 that this doesn't crash (it shouldn't — an empty batch list just means "nothing found," a non-fatal note) and doesn't change that test's existing count assertions.

---

## File Structure

- `backend/app/services/integrations/roboflow_search.py` — **create**: the shared, retry-hardened `/search` wrapper, extracted from `roboflow_import.py` so `roboflow_export.py` can reuse it without duplicating the retry/error-envelope logic.
- `backend/app/services/integrations/roboflow_import.py` — **modify**: remove the now-duplicated `_rf_search_page`/`_describe_unreachable`/search-retry constants, import them from the new module instead.
- `backend/app/services/integrations/roboflow_export.py` — **modify**: the bulk of the new logic — `_discover_batch_membership`, the replacement for `_assign_annotating_review_job` (`_move_to_annotating`), and the small addition to `push_version_to_roboflow`'s main loop that collects which known-id images actually got updated.
- `backend/tests/test_roboflow_jobs.py` — **modify**: extend `_FakeRoboflowProject` with `create_annotation_batch`/`merge_annotation_batches`, add new tests for the carve/merge/job-creation logic and its failure modes.

---

### Task 1: Extract the shared Roboflow `/search` helper

**Files:**
- Create: `backend/app/services/integrations/roboflow_search.py`
- Modify: `backend/app/services/integrations/roboflow_import.py:24-53` (imports/module-level constants), `:128-193` (the retry constants + `_describe_unreachable` + `_rf_search_page` definitions), `:616` (the call site)
- Test: no new test file — this is a pure refactor (same logic, same call site, same signature) and the existing `test_roboflow_jobs.py` import tests (`test_roboflow_import_job_raw_pull_when_no_version` and friends, which exercise `_rf_search_page` indirectly via `_fake_search_post`/`monkeypatch.setattr(roboflow_import_module.requests, "post", ...)`) are the regression coverage — they must still pass unmodified after this move.

**Interfaces:**
- Produces: `roboflow_search.rf_search_page(rf_project, api_key: str, *, offset: int, limit: int, fields: list[str], batch_id: str | None = None) -> list[dict]` — same behavior as the old `roboflow_import._rf_search_page`, just importable from elsewhere. Consumed by Task 2 (`roboflow_export.py`'s new batch-discovery code) and by `roboflow_import.py`'s existing call site (updated in this task).

- [ ] **Step 1: Create the new module**

Create `backend/app/services/integrations/roboflow_search.py`:

```python
"""Shared, retry-hardened wrapper around Roboflow's `/search` endpoint.
Originally lived only in `roboflow_import.py` (pulling images in); moved
here once `roboflow_export.py` needed the identical retry/error-envelope
handling to discover which batch an already-known image currently belongs
to, before carving it into a review job (see
`roboflow_export._discover_batch_membership`)."""
from __future__ import annotations

import logging
import time

import requests

logger = logging.getLogger(__name__)

# Roboflow's /search occasionally throws a transient 5xx (observed: bare
# HTTP 500 "contact support" bodies for a few minutes at a time) or a 429.
# Without a retry, one blip fails the whole multi-page caller. Retry those
# statuses and connection/timeout errors with exponential backoff; a 4xx or
# an {"error": ...} envelope is not transient and still raises at once.
# Backoff between the 4 attempts: 1s, 2s, 4s.
_SEARCH_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_SEARCH_MAX_ATTEMPTS = 4
_SEARCH_BACKOFF_BASE_S = 1.0


def _describe_unreachable(exc: requests.RequestException, attempts: int) -> str:
    """A `requests.exceptions.ConnectionError` (which is what a DNS
    resolution failure like `NameResolutionError` surfaces as) means the
    request never reached Roboflow at all — that's a local network/DNS/
    firewall/proxy problem on this machine, not "Roboflow is having a bad
    moment", and telling the user to just retry in a few minutes is wrong
    advice for it (observed live: users read that hint, wait, and hit the
    exact same DNS failure again). Anything else (timeouts, etc.) keeps the
    original "transient, retry later" framing, which is accurate for those."""
    if isinstance(exc, requests.exceptions.ConnectionError):
        return (
            f"Could not reach Roboflow (api.roboflow.com) after {attempts} attempts — this "
            "machine was unable to connect at all, which usually means a local internet, DNS, "
            f"or firewall/proxy problem rather than an issue on Roboflow's side. ({exc})"
        )
    return (
        f"Roboflow /search could not be reached after {attempts} attempts ({exc}). This is "
        "usually a temporary Roboflow-side issue — retry in a few minutes."
    )


def rf_search_page(
    rf_project,
    api_key: str,
    *,
    offset: int,
    limit: int,
    fields: list[str],
    batch_id: str | None = None,
) -> list[dict]:
    """Direct call to Roboflow's `/search` endpoint, in place of
    `rf_project.search()`. The SDK (roboflow==1.4.1) ends `search()` with a
    bare `data.json()["results"]` — no status check, no error-envelope
    check (unlike `Workspace.create_project()` right beside it, which does
    `if "error" in r.json()`) — so any response that isn't
    `{"results": [...]}` (an `{"error": ...}` body, a changed response
    shape, a permission failure on this one endpoint) surfaces only as an
    opaque `KeyError: 'results'` from deep in the SDK with the real cause
    swallowed. This issues the identical request (same URL and payload
    shape the SDK builds from `rf_project.id`) but inspects the response,
    retries a transient 5xx/429, and keeps the body in both the raised
    error and the logs.

    `batch_id`, when given, narrows results to that one upload batch —
    same as the SDK's `search(batch=True, batch_id=...)`. Note: Roboflow's
    valid `fields` list does NOT include anything that reports which batch
    an image belongs to (confirmed live: `"Invalid fields requested: ...
    Valid fields are: annotations, aspectRatio, created, embedding,
    features, filename, height, id, labels, name, owner, projects, score,
    split, tags, url, user_metadata, width, offset, limit"`) — the only way
    to learn "is image X in batch Y" is to ask for batch Y's own image list
    (via `batch_id`) and check."""
    from roboflow.config import API_URL

    url = f"{API_URL}/{rf_project.id}/search?api_key={api_key}"
    payload = {
        "offset": offset,
        "limit": limit,
        "batch": batch_id is not None,
        "fields": fields,
    }
    if batch_id is not None:
        payload["batch_id"] = batch_id

    for attempt in range(1, _SEARCH_MAX_ATTEMPTS + 1):
        try:
            resp = requests.post(url, json=payload, timeout=30)
        except requests.RequestException as exc:
            logger.warning(
                "Roboflow /search request error (attempt %d/%d): %s",
                attempt,
                _SEARCH_MAX_ATTEMPTS,
                exc,
            )
            if attempt == _SEARCH_MAX_ATTEMPTS:
                raise RuntimeError(_describe_unreachable(exc, _SEARCH_MAX_ATTEMPTS)) from exc
            time.sleep(_SEARCH_BACKOFF_BASE_S * 2 ** (attempt - 1))
            continue

        if resp.status_code in _SEARCH_RETRY_STATUSES and attempt < _SEARCH_MAX_ATTEMPTS:
            logger.warning(
                "Roboflow /search returned HTTP %s (attempt %d/%d) — retrying",
                resp.status_code,
                attempt,
                _SEARCH_MAX_ATTEMPTS,
            )
            time.sleep(_SEARCH_BACKOFF_BASE_S * 2 ** (attempt - 1))
            continue

        break

    try:
        body = resp.json()
    except ValueError as exc:
        logger.error(
            "Roboflow /search returned non-JSON (HTTP %s): %s", resp.status_code, resp.text[:2000]
        )
        raise RuntimeError(
            f"Roboflow /search returned a non-JSON response (HTTP {resp.status_code})"
        ) from exc

    if resp.status_code != 200 or not isinstance(body, dict) or "results" not in body:
        logger.error("Roboflow /search failed (HTTP %s): %s", resp.status_code, body)
        detail = (body.get("error") or body.get("message") or body) if isinstance(body, dict) else body
        hint = (
            " This is usually a temporary Roboflow-side issue — retry in a few minutes."
            if resp.status_code in _SEARCH_RETRY_STATUSES
            else ""
        )
        raise RuntimeError(f"Roboflow /search failed (HTTP {resp.status_code}): {detail}{hint}")

    return body["results"]
```

(This is `roboflow_import.py`'s current `_describe_unreachable` + `_rf_search_page`, verbatim except: function renamed `_rf_search_page` → `rf_search_page` (public — it's meant to be imported now), and the two `"retry the import"` phrasings that were import-specific are generalized to `"retry in a few minutes"` since this helper no longer belongs to the import path alone.)

- [ ] **Step 2: Update `roboflow_import.py` to use the shared module**

Remove from `backend/app/services/integrations/roboflow_import.py`:
- The three constants at lines 134-136 (`_SEARCH_RETRY_STATUSES`, `_SEARCH_MAX_ATTEMPTS`, `_SEARCH_BACKOFF_BASE_S`) and their preceding comment block (lines 128-133).
- The `_describe_unreachable` function (lines 175-193).
- The `_rf_search_page` function (lines 421-505, i.e. everything from `def _rf_search_page(` through its final `return body["results"]`).

Add an import near the top of the file (with the other `from app.services...` imports, e.g. after `from app.services.annotation.service import create_annotation`):

```python
from app.services.integrations.roboflow_search import rf_search_page
```

Update the one call site (around line 616):

```python
            page = _rf_search_page(
```

becomes:

```python
            page = rf_search_page(
```

Run a final check that nothing else in the file references the removed names:

```bash
grep -n "_rf_search_page\|_describe_unreachable\|_SEARCH_RETRY_STATUSES\|_SEARCH_MAX_ATTEMPTS\|_SEARCH_BACKOFF_BASE_S" backend/app/services/integrations/roboflow_import.py
```

Expected: no output (every reference was either removed or was the one call site just updated).

- [ ] **Step 3: Run the existing import test suite to confirm the refactor is behavior-preserving**

Run: `cd backend && ./venv/Scripts/python.exe -m pytest tests/test_roboflow_jobs.py -k import -v`
Expected: all PASS, identical to before this task (these tests monkeypatch the shared `requests` module object itself — `monkeypatch.setattr(roboflow_import_module.requests, "post", _fake_search_post)` — which patches the single global `requests.post` regardless of which module calls it, so moving `rf_search_page` to a new module does not require any test changes).

Then run the full file once: `./venv/Scripts/python.exe -m pytest tests/test_roboflow_jobs.py -q`
Expected: all PASS (no regressions — nothing else in this file touches these names).

- [ ] **Step 4: Commit**

```bash
git add backend/app/services/integrations/roboflow_search.py backend/app/services/integrations/roboflow_import.py
git commit -m "refactor(roboflow): extract shared /search helper into roboflow_search.py"
```

---

### Task 2: Batch-membership discovery for export

**Files:**
- Modify: `backend/app/services/integrations/roboflow_export.py` (add `_discover_batch_membership` and its page-size constant, near the other export-side helpers — e.g. right after `_is_duplicate_upload`, before `_upload_one_image`)
- Test: `backend/tests/test_roboflow_jobs.py`

**Interfaces:**
- Consumes: `roboflow_search.rf_search_page` (Task 1).
- Produces: `_discover_batch_membership(project, api_key: str, *, target_ids: set[str]) -> dict[str, list[str]]` — consumed by Task 3's `_move_to_annotating`.

- [ ] **Step 1: Write the failing tests**

Add to `backend/tests/test_roboflow_jobs.py` (near the other `test_push_version_*` tests — this is a standalone unit test of the pure function, no HTTP/DB involved):

```python
def test_discover_batch_membership_finds_targets_across_multiple_batches(monkeypatch) -> None:
    """Two target ids live in two different batches; a third batch (with
    nothing we want) must still get scanned if the earlier batches didn't
    account for every target, but scanning stops as soon as all targets
    are found — verified by asserting the third batch's search is never
    called when the first two already account for everything."""
    from app.services.integrations import roboflow_export as mod

    class _Proj:
        id = "ws/proj"

        def get_batches(self):
            return {
                "batches": [
                    {"id": "batch-a", "images": 2},
                    {"id": "batch-b", "images": 2},
                    {"id": "batch-c", "images": 5},
                ]
            }

    search_calls = []

    def _fake_search_page(project, api_key, *, offset, limit, fields, batch_id=None):
        search_calls.append(batch_id)
        if batch_id == "batch-a":
            return [{"id": "img-1"}, {"id": "img-other-a"}]
        if batch_id == "batch-b":
            return [{"id": "img-2"}, {"id": "img-other-b"}]
        raise AssertionError(f"batch-c should never be scanned, all targets already found; got {batch_id!r}")

    monkeypatch.setattr(mod, "rf_search_page", _fake_search_page)

    result = mod._discover_batch_membership(_Proj(), "fake-key", target_ids={"img-1", "img-2"})

    assert result == {"batch-a": ["img-1"], "batch-b": ["img-2"]}
    assert search_calls == ["batch-a", "batch-b"]


def test_discover_batch_membership_paginates_within_one_batch(monkeypatch) -> None:
    """A target id on page 2 of a single large batch is still found —
    pagination within one batch must continue until the target turns up
    or the batch is exhausted."""
    from app.services.integrations import roboflow_export as mod

    class _Proj:
        id = "ws/proj"

        def get_batches(self):
            return {"batches": [{"id": "batch-a", "images": 3}]}

    pages = {
        0: [{"id": "img-x"}, {"id": "img-y"}],
        2: [{"id": "img-z"}],
    }

    def _fake_search_page(project, api_key, *, offset, limit, fields, batch_id=None):
        return pages.get(offset, [])

    monkeypatch.setattr(mod, "rf_search_page", _fake_search_page)
    monkeypatch.setattr(mod, "_BATCH_SCAN_PAGE_SIZE", 2)

    result = mod._discover_batch_membership(_Proj(), "fake-key", target_ids={"img-z"})

    assert result == {"batch-a": ["img-z"]}


def test_discover_batch_membership_returns_empty_for_ids_never_found() -> None:
    """No crash, no exception — an id that isn't in any batch (deleted on
    Roboflow, or genuinely never belonged to a batch) is simply absent
    from the returned mapping; the caller decides what that means."""
    from app.services.integrations import roboflow_export as mod

    class _Proj:
        id = "ws/proj"

        def get_batches(self):
            return {"batches": []}

    result = mod._discover_batch_membership(_Proj(), "fake-key", target_ids={"img-missing"})

    assert result == {}


def test_discover_batch_membership_empty_target_set_short_circuits(monkeypatch) -> None:
    """No targets means no work — `get_batches()` must never even be
    called."""
    from app.services.integrations import roboflow_export as mod

    class _Proj:
        def get_batches(self):
            raise AssertionError("get_batches() should never be called for an empty target set")

    result = mod._discover_batch_membership(_Proj(), "fake-key", target_ids=set())

    assert result == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && ./venv/Scripts/python.exe -m pytest tests/test_roboflow_jobs.py -k discover_batch_membership -v`
Expected: FAIL — `AttributeError: module 'app.services.integrations.roboflow_export' has no attribute '_discover_batch_membership'` (and no `rf_search_page` name to monkeypatch yet either).

- [ ] **Step 3: Implement**

In `backend/app/services/integrations/roboflow_export.py`, add the import (with the existing imports, e.g. right after `from app.services.integrations.roboflow_connect import get_client`):

```python
from app.services.integrations.roboflow_search import rf_search_page
```

Add near `_UPLOAD_MAX_ATTEMPTS`/`_EXPORT_MAX_WORKERS` (the other tuning constants):

```python
# Same page size `roboflow_import._RAW_SEARCH_PAGE_SIZE` uses for the
# identical shape of paginated /search call.
_BATCH_SCAN_PAGE_SIZE = 100
```

Add the function itself, right after `_is_duplicate_upload` (before `_upload_one_image`):

```python
def _discover_batch_membership(project, api_key: str, *, target_ids: set[str]) -> dict[str, list[str]]:
    """Returns `{batch_id: [image_id, ...]}` for as much of `target_ids` as
    could be located, by paging through the project's existing batches
    until every target id has turned up or every batch has been scanned.
    Roboflow's `/search` has no "which batch is image X in" field and no
    "look up these specific ids" filter (confirmed live — see
    `roboflow_search.rf_search_page`'s docstring) — narrowing to one batch
    at a time via `batch_id` and checking each page's ids against the
    target set is the only way to find out. Stops scanning entirely, even
    mid-batch, once every target id has been located — the common case
    (a handful of just-updated images) shouldn't cost a full scan of every
    batch in a large project."""
    remaining = set(target_ids)
    found: dict[str, list[str]] = {}
    if not remaining:
        return found

    batches = project.get_batches().get("batches", [])
    for batch in batches:
        if not remaining:
            break
        batch_id = batch.get("id")
        if not batch_id or not batch.get("images"):
            continue
        offset = 0
        while remaining:
            page = rf_search_page(
                project, api_key, offset=offset, limit=_BATCH_SCAN_PAGE_SIZE, fields=["id"], batch_id=batch_id
            )
            if not page:
                break
            for item in page:
                item_id = item.get("id")
                if item_id in remaining:
                    found.setdefault(batch_id, []).append(item_id)
                    remaining.discard(item_id)
            if len(page) < _BATCH_SCAN_PAGE_SIZE:
                break
            offset += _BATCH_SCAN_PAGE_SIZE
    return found
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && ./venv/Scripts/python.exe -m pytest tests/test_roboflow_jobs.py -k discover_batch_membership -v`
Expected: all 4 PASS.

Then the full file: `./venv/Scripts/python.exe -m pytest tests/test_roboflow_jobs.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/integrations/roboflow_export.py backend/tests/test_roboflow_jobs.py
git commit -m "feat(roboflow): discover which batch an already-known image currently belongs to"
```

---

### Task 3: Carve annotation-updated images into Annotating alongside new images

**Files:**
- Modify: `backend/app/services/integrations/roboflow_export.py` (replace `_assign_annotating_review_job` with `_move_to_annotating`; add the small collection step in `push_version_to_roboflow`'s main loop; rewire the tail block)
- Modify: `backend/tests/test_roboflow_jobs.py` (extend `_FakeRoboflowProject`; verify/adjust `test_push_version_known_roboflow_id_routes_through_save_annotation_not_upload`; add new coverage)

**Interfaces:**
- Consumes: `_discover_batch_membership` (Task 2), `_PushOutcome`/`PushResult` (already in this file, from the prior branch).
- Produces: `_move_to_annotating(project, api_key: str, *, batch_name: str, labeler_email: str | None, new_images_uploaded: bool, updated_roboflow_ids: list[str]) -> str | None` — replaces `_assign_annotating_review_job` at the one call site inside `push_version_to_roboflow`.

- [ ] **Step 1: Extend the test double**

In `backend/tests/test_roboflow_jobs.py`, `_FakeRoboflowProject` currently has `get_batches`/`create_annotation_job`/`upload`/`save_annotation` (with `self.uploads`/`self.annotation_jobs`/`self.saved_annotations` tracked in `__init__`). `get_batches()`'s docstring comment currently says `` `_assign_annotating_review_job` can find the real batch a test's export just created `` — update that one name to `` `_move_to_annotating` `` since this task renames the function.

Add two more methods, right after `create_annotation_job` (i.e. right after its `return {"id": "job-1"}`, before `def upload`):

```python
    def create_annotation_batch(self, *, source_batch_id: str, image_ids: list[str], name: str | None = None) -> dict:
        self.carved_batches.append(
            {"source_batch_id": source_batch_id, "image_ids": list(image_ids), "name": name}
        )
        new_id = f"carved-{len(self.carved_batches)}"
        return {"batchId": new_id, "movedImageCount": len(image_ids)}

    def merge_annotation_batches(self, *, source_batch_ids: list[str], target_batch_id: str) -> dict:
        self.merged_batches.append({"source_batch_ids": list(source_batch_ids), "target_batch_id": target_batch_id})
        return {}
```

And add the two new list attributes in `__init__`, alongside the existing `self.uploads`/`self.annotation_jobs`/`self.saved_annotations`:

```python
        self.carved_batches: list[dict] = []
        self.merged_batches: list[dict] = []
```

- [ ] **Step 2: Write the failing tests**

Add to `backend/tests/test_roboflow_jobs.py`:

```python
def test_push_version_updates_move_known_id_images_to_new_batch_and_job(
    real_db_session, monkeypatch, unique_name: str
) -> None:
    """Core of Task 3: a known-Roboflow-id image whose annotation gets
    updated must be carved out of its current Roboflow batch into a new
    one named after this export, and a review job filed for it — even
    though there were zero new images (the exact all-existing-images
    scenario this whole feature exists for)."""
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

    class _Proj:
        def save_annotation(self, **kwargs):
            return ({"success": True}, 0.0, 0)

        def get_batches(self):
            return {"batches": [{"id": "original-batch", "images": 1}]}

        def create_annotation_batch(self, *, source_batch_id, image_ids, name=None):
            assert source_batch_id == "original-batch"
            assert image_ids == ["known-rf-image-id"]
            assert name == "custom-review-batch"
            return {"batchId": "carved-batch-1", "movedImageCount": 1}

        def create_annotation_job(self, **kwargs):
            assert kwargs["batch_id"] == "carved-batch-1"
            assert kwargs["name"] == "custom-review-batch"
            assert kwargs["labeler_email"] == "a@b.com"
            assert kwargs["reviewer_email"] == "a@b.com"
            return {"id": "job-1"}

    calls = {"n": 0}

    def _fake_search_page(project, api_key, *, offset, limit, fields, batch_id=None):
        calls["n"] += 1
        assert batch_id == "original-batch"
        return [{"id": "known-rf-image-id"}]

    monkeypatch.setattr(mod, "rf_search_page", _fake_search_page)

    class _WS:
        def project(self, slug):
            return _Proj()

    class _RF:
        def workspace(self, ws):
            return _WS()

    monkeypatch.setattr(mod, "get_client", lambda db: (_RF(), {"default_labeler_email": "a@b.com", "api_key": "fake-key"}))

    result = mod.push_version_to_roboflow(
        real_db_session,
        version_id=_uuid.uuid4(),
        workspace="ws",
        project_slug="proj",
        custom_batch_name="custom-review-batch",
    )

    assert (result.new_images, result.annotations_updated, result.failed) == (0, 1, 0)
    assert calls["n"] == 1


def test_move_to_annotating_merges_new_and_updated_into_one_job(monkeypatch) -> None:
    """A mixed push (some genuinely new images, some existing images whose
    annotation was updated) must end up as ONE review job covering both —
    not two separate, confusingly-named batches."""
    from app.services.integrations import roboflow_export as mod

    class _Proj:
        def get_batches(self):
            return {"batches": [{"id": "new-image-batch", "name": "my-batch", "images": 3}]}

        def create_annotation_batch(self, *, source_batch_id, image_ids, name=None):
            return {"batchId": "carved-batch-1", "movedImageCount": len(image_ids)}

        def merge_annotation_batches(self, *, source_batch_ids, target_batch_id):
            merge_calls.append((source_batch_ids, target_batch_id))
            return {}

        def create_annotation_job(self, **kwargs):
            job_calls.append(kwargs)
            return {"id": "job-1"}

    merge_calls: list[tuple] = []
    job_calls: list[dict] = []

    def _fake_search_page(project, api_key, *, offset, limit, fields, batch_id=None):
        return [{"id": "known-rf-image-id"}]

    monkeypatch.setattr(mod, "rf_search_page", _fake_search_page)

    note = mod._move_to_annotating(
        _Proj(),
        "fake-key",
        batch_name="my-batch",
        labeler_email="a@b.com",
        new_images_uploaded=True,
        updated_roboflow_ids=["known-rf-image-id"],
    )

    assert note is None
    assert merge_calls == [(["carved-batch-1"], "new-image-batch")]
    assert job_calls == [
        {"name": "my-batch", "batch_id": "new-image-batch", "labeler_email": "a@b.com", "reviewer_email": "a@b.com"}
    ]


def test_move_to_annotating_carves_multiple_source_batches_and_merges_them(monkeypatch) -> None:
    """Updated images spanning two different original Roboflow batches
    each get carved out separately, then the two resulting batches are
    merged into one before filing a single job."""
    from app.services.integrations import roboflow_export as mod

    class _Proj:
        def get_batches(self):
            return {"batches": [{"id": "batch-a", "images": 5}, {"id": "batch-b", "images": 5}]}

        def create_annotation_batch(self, *, source_batch_id, image_ids, name=None):
            carve_calls.append((source_batch_id, image_ids))
            return {"batchId": f"carved-{source_batch_id}", "movedImageCount": len(image_ids)}

        def merge_annotation_batches(self, *, source_batch_ids, target_batch_id):
            merge_calls.append((sorted(source_batch_ids), target_batch_id))
            return {}

        def create_annotation_job(self, **kwargs):
            job_calls.append(kwargs)
            return {"id": "job-1"}

    carve_calls: list[tuple] = []
    merge_calls: list[tuple] = []
    job_calls: list[dict] = []

    def _fake_search_page(project, api_key, *, offset, limit, fields, batch_id=None):
        if batch_id == "batch-a":
            return [{"id": "img-1"}]
        return [{"id": "img-2"}]

    monkeypatch.setattr(mod, "rf_search_page", _fake_search_page)

    note = mod._move_to_annotating(
        _Proj(),
        "fake-key",
        batch_name="my-batch",
        labeler_email="a@b.com",
        new_images_uploaded=False,
        updated_roboflow_ids=["img-1", "img-2"],
    )

    assert note is None
    assert sorted(carve_calls) == [("batch-a", ["img-1"]), ("batch-b", ["img-2"])]
    assert len(merge_calls) == 1
    assert len(job_calls) == 1


def test_move_to_annotating_reports_images_not_found_in_any_batch(monkeypatch) -> None:
    """An updated image that can't be located in any Roboflow batch (e.g.
    deleted there since the annotation update) must not crash the export —
    it's reported in the note and simply stays wherever it is."""
    from app.services.integrations import roboflow_export as mod

    class _Proj:
        def get_batches(self):
            return {"batches": []}

    note = mod._move_to_annotating(
        _Proj(),
        "fake-key",
        batch_name="my-batch",
        labeler_email="a@b.com",
        new_images_uploaded=False,
        updated_roboflow_ids=["img-missing"],
    )

    assert note is not None
    assert "1" in note and "could not be located" in note


def test_move_to_annotating_no_work_returns_none(monkeypatch) -> None:
    """Neither new images nor updated ones — nothing to do, no API calls,
    no note."""
    from app.services.integrations import roboflow_export as mod

    class _Proj:
        def get_batches(self):
            raise AssertionError("must not be called when there's nothing to move")

    note = mod._move_to_annotating(
        _Proj(),
        "fake-key",
        batch_name="my-batch",
        labeler_email="a@b.com",
        new_images_uploaded=False,
        updated_roboflow_ids=[],
    )

    assert note is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd backend && ./venv/Scripts/python.exe -m pytest tests/test_roboflow_jobs.py -k "move_to_annotating or updates_move_known_id" -v`
Expected: FAIL — `AttributeError: module ... has no attribute '_move_to_annotating'`, and the `_FakeRoboflowProject.create_annotation_batch`/`merge_annotation_batches` methods don't exist yet either (fix Step 1 first if you haven't already applied it before running this).

- [ ] **Step 4: Implement `_move_to_annotating`**

In `backend/app/services/integrations/roboflow_export.py`, **replace** the entire `_assign_annotating_review_job` function (currently lines 209-255) with:

```python
def _move_to_annotating(
    project,
    api_key: str,
    *,
    batch_name: str,
    labeler_email: str | None,
    new_images_uploaded: bool,
    updated_roboflow_ids: list[str],
) -> str | None:
    """Moves this export's images out of Roboflow's Unassigned column into
    Annotating, covering two disjoint sources that need different Roboflow
    API calls to get there:

    - Genuinely new images (`new_images_uploaded`) already landed together
      in a batch named `batch_name` at upload time (uploading with
      `is_prediction=True` only gets an image as far as Unassigned —
      moving it into Annotating needs the separate
      `Project.create_annotation_job` call below) — just needs that call
      pointed at the batch Roboflow already grouped them into.
    - Already-existing images whose annotation this export just updated
      via the known-Roboflow-id path (`updated_roboflow_ids` — see
      `_save_annotation_only`; the SDK-duplicate fallback path's updates
      are deliberately NOT included here, since discovering their
      Roboflow ids would need extra plumbing for what's already the rarer
      case) are scattered across whatever batch(es) they already belonged
      to. There is no "create a job for these arbitrary existing images"
      call — only `Project.create_annotation_batch(source_batch_id,
      image_ids)`, which carves a named subset OUT of a batch it's
      already in (confirmed live against a real Roboflow project: carving
      2 images out of an 11-image batch left the other 9 untouched and
      created a new batch with exactly those 2). Each involved source
      batch (found via `_discover_batch_membership`) gets its own carve
      call.

    If both sources produced a batch, or the updated-image carve touched
    more than one source batch, the results are merged into one via
    `Project.merge_annotation_batches` before the single
    `Project.create_annotation_job` call — one job per export, covering
    everything that needs review, under one name.

    Returns `None` on full success (or when there's nothing to do), or a
    short message describing what went wrong, for the caller to fold into
    the export's summary note non-fatally — the images/annotations
    themselves already landed on Roboflow either way; this call only
    affects which Annotate-board column they sit in."""
    if not new_images_uploaded and not updated_roboflow_ids:
        return None
    if not labeler_email:
        return (
            "Images uploaded, but no default labeler/reviewer email is set (Settings -> Roboflow) — "
            "they're sitting in Roboflow's Unassigned column instead of Annotating. Set that email and "
            "re-export, or create the review job yourself in Roboflow's Annotate tab."
        )

    candidate_batch_ids: list[str] = []
    notes: list[str] = []

    if new_images_uploaded:
        try:
            batches = project.get_batches().get("batches", [])
            batch = next((b for b in batches if b.get("name") == batch_name), None)
            if batch is None:
                notes.append(
                    f"couldn't find batch {batch_name!r} on Roboflow to move the newly uploaded images into "
                    "Annotating — they're sitting in Unassigned."
                )
            else:
                candidate_batch_ids.append(batch["id"])
        except Exception as exc:  # noqa: BLE001 — non-fatal: images already uploaded
            notes.append(f"couldn't look up the new-image batch on Roboflow ({exc}) — it's sitting in Unassigned.")

    if updated_roboflow_ids:
        membership = _discover_batch_membership(project, api_key, target_ids=set(updated_roboflow_ids))
        for source_batch_id, image_ids in membership.items():
            try:
                carved = project.create_annotation_batch(
                    source_batch_id=source_batch_id, image_ids=image_ids, name=batch_name
                )
                candidate_batch_ids.append(carved["batchId"])
            except Exception as exc:  # noqa: BLE001 — non-fatal: annotations already updated
                notes.append(f"couldn't move {len(image_ids)} updated image(s) out of Unassigned ({exc}).")
        not_found = len(updated_roboflow_ids) - sum(len(v) for v in membership.values())
        if not_found > 0:
            notes.append(
                f"{not_found} updated image(s) could not be located in any Roboflow batch and stayed in "
                "Unassigned."
            )

    if not candidate_batch_ids:
        return " ".join(notes) if notes else None

    target_batch_id = candidate_batch_ids[0]
    if len(candidate_batch_ids) > 1:
        try:
            project.merge_annotation_batches(
                source_batch_ids=candidate_batch_ids[1:], target_batch_id=target_batch_id
            )
        except Exception as exc:  # noqa: BLE001 — non-fatal, still try to file a job for the first batch below
            notes.append(f"couldn't combine new and updated images into one review job ({exc}).")

    try:
        project.create_annotation_job(
            name=batch_name,
            batch_id=target_batch_id,
            labeler_email=labeler_email,
            reviewer_email=labeler_email,
        )
    except Exception as exc:  # noqa: BLE001 — non-fatal: images/annotations already landed
        logger.warning("Roboflow export: could not auto-create annotation job for batch %r: %s", batch_name, exc)
        notes.append(
            f"auto-creating the Roboflow review job failed ({exc}) — check {labeler_email!r} is a member of "
            "this Roboflow workspace, or create the job yourself in Roboflow's Annotate tab."
        )

    return " ".join(notes) if notes else None
```

- [ ] **Step 5: Collect updated-known-image ids and rewire the tail block**

Still in `push_version_to_roboflow`, add a new collector alongside the existing counters. Find:

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

Replace with:

```python
        new_images = 0
        annotations_updated = 0
        unchanged = 0
        succeeded = 0
        failed = 0
        failures: list[str] = []
        # Only ever appended to for the known-Roboflow-id path (see
        # `_move_to_annotating`'s docstring on the scope boundary) — an
        # `ANNOTATION_UPDATED` outcome from the SDK-duplicate fallback path
        # has no entry in `known_roboflow_ids` and is deliberately skipped.
        updated_known_roboflow_ids: list[str] = []
        seen_statuses: list[int | None] = []
        fail_fast_error: RoboflowExportError | None = None
```

Then find the per-completion outcome handling:

```python
                        outcome = future.result()
                        succeeded += 1
                        if outcome is _PushOutcome.NEW_IMAGE:
                            new_images += 1
                        elif outcome is _PushOutcome.ANNOTATION_UPDATED:
                            annotations_updated += 1
                        else:
                            unchanged += 1
```

Replace with:

```python
                        outcome = future.result()
                        succeeded += 1
                        if outcome is _PushOutcome.NEW_IMAGE:
                            new_images += 1
                        elif outcome is _PushOutcome.ANNOTATION_UPDATED:
                            annotations_updated += 1
                            known_id = known_roboflow_ids.get(_image_uuid(image_path))
                            if known_id is not None:
                                updated_known_roboflow_ids.append(known_id)
                        else:
                            unchanged += 1
```

Finally, replace the tail block's review-job section. Find:

```python
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
```

Replace with:

```python
    review_job_note: str | None = None
    if upload_target == RoboflowUploadTarget.ANNOTATING.value and (new_images > 0 or updated_known_roboflow_ids):
        review_job_note = _move_to_annotating(
            project,
            config["api_key"],
            batch_name=batch_name,
            labeler_email=config.get("default_labeler_email"),
            new_images_uploaded=new_images > 0,
            updated_roboflow_ids=updated_known_roboflow_ids,
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd backend && ./venv/Scripts/python.exe -m pytest tests/test_roboflow_jobs.py -k "move_to_annotating or updates_move_known_id" -v`
Expected: all PASS.

Then run the specific pre-existing tests this task's Global Constraints section named as needing to stay green: `./venv/Scripts/python.exe -m pytest tests/test_roboflow_jobs.py -k "annotating_without_labeler_email or annotating_creates_review_job or dataset_target_never_creates or all_duplicates_updates_annotations or mixed_new_and_duplicate or known_roboflow_id_routes_through" -v`
Expected: all PASS. If `test_push_version_known_roboflow_id_routes_through_save_annotation_not_upload` fails, read why carefully before changing it — per this plan's Global Constraints, it's expected to now also exercise `_move_to_annotating` (since it has exactly one known-id `ANNOTATION_UPDATED` outcome and `default_labeler_email` set), which will call `project.get_batches()` on its existing `_Proj` fake (already returns `{"batches": []}`) and produce a non-crashing "not located" note — this should not break its existing count assertions (`result.new_images`, `result.annotations_updated`, etc.), only add an unasserted `result.note` value. If it does fail for a different reason, fix the actual cause rather than loosening the assertion.

Then the full suite: `./venv/Scripts/python.exe -m pytest tests/ -q`
Expected: all PASS, pristine output.

- [ ] **Step 7: Update `push_version_to_roboflow`'s docstring**

Its docstring still says `` `ANNOTATING` also tries to move the pushed batch into Roboflow's "Annotating" column via a second API call once uploads finish — see `_assign_annotating_review_job` ``. Update that sentence to:

```python
    `ANNOTATING` also tries to move newly uploaded images, and any
    existing images whose annotation this export updated via the known-
    Roboflow-id path, into Roboflow's "Annotating" column via a second
    round of API calls once uploads finish — see `_move_to_annotating` —
    which needs a default labeler email configured (Settings -> Roboflow);
```

(replacing the corresponding lines in the existing docstring paragraph, keeping the rest of that paragraph — "its failure (or that email being unset) is folded into the returned `PushResult.note` but never fails the export, since the images themselves already uploaded fine." — unchanged).

- [ ] **Step 8: Commit**

```bash
git add backend/app/services/integrations/roboflow_export.py backend/tests/test_roboflow_jobs.py
git commit -m "feat(roboflow): move annotation-updated known-id images into Annotating

Previously only genuinely new images got moved out of Unassigned into a
review job. Images already in the target Roboflow project whose
annotation this export just updated (via the known-Roboflow-id direct
path) now get carved into their own named batch and job too — merged
with the new-image batch into one job when both exist in the same push.
Discovering which batch an existing image currently belongs to (Roboflow
exposes no such field) is done by paging each of the project's batches
via /search and checking for the target ids, stopping once every target
is found."
```

---

## Self-Review Notes (already applied above, recorded for the record)

- **Spec coverage:** trigger is automatic (Task 3, gated the same way as the existing new-image path — no new flag/parameter added anywhere in the call chain, confirmed no opt-in mechanism was introduced). Naming reuses `batch_name` (Task 3's `_move_to_annotating` call passes `batch_name=batch_name` for both the new-image lookup and the carve/job calls — one name throughout, no suffix).
- **Type consistency:** `_discover_batch_membership`'s signature (Task 2) matches exactly how Task 3 calls it (`project, api_key, target_ids=set(...)`  → `dict[str, list[str]]`, unpacked via `.items()`). `_move_to_annotating`'s signature matches its one call site in `push_version_to_roboflow` exactly (same keyword names: `batch_name`, `labeler_email`, `new_images_uploaded`, `updated_roboflow_ids`).
- **No placeholders:** every step above contains the literal code to write; no "add error handling"-style steps.
