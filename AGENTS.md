# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this is

A human-in-the-loop computer-vision annotation tool built around one loop:

```
existing model → auto-annotate new footage → human corrects → versioned
dataset → train a new model → register it → it becomes the new annotator
```

The taxonomy is never hardcoded — the app reads a model's class list from its own weights. Validated against a real Ultralytics YOLO detector (`detect_v1.pt`) with a documented failure mode (a player's foot mistaken for a cone, confidence 0.336) that the quality-analysis layer is designed to catch. See `frontend/DESIGN.md` and `backend/app/services/quality/` for specifics, and `docs/WORKFLOW.md` for the six-stage loop as experienced by a user (Import → Auto-annotate → Review → Version → Train → New model), including which sidebar page does what and why a stage might be blocked.

## Commands

Run everything: `docker compose up --build` (Postgres, Redis, MinIO, API, Celery worker, frontend all start together; backend runs `alembic upgrade head` on startup).

Faster local iteration (infra in Docker, app processes native):

```bash
docker compose up -d postgres redis minio

cd backend
./venv/Scripts/python -m alembic upgrade head
./venv/Scripts/python -m uvicorn app.main:app --reload
# separately — background jobs (video extraction, auto-annotation, training, quality):
./venv/Scripts/python -m celery -A app.workers.celery_app worker -B -Q gpu,default -c 1 --pool=solo  # --pool=solo required on Windows
```

```bash
cd frontend
npm run dev
```

Backend tests (needs Postgres+Redis reachable — `docker compose up -d postgres redis minio`):

```bash
cd backend
DATABASE_URL=postgresql+psycopg://annotate:annotate@localhost:5432/annotate_test ./venv/Scripts/python -m pytest tests/ -q
# single test:
DATABASE_URL=postgresql+psycopg://annotate:annotate@localhost:5432/annotate_test ./venv/Scripts/python -m pytest tests/test_quality_analyzer.py::test_name -q
```

Tests run fully offline against fake model objects, except `test_real_model_integration.py`, which loads real weights from `artifacts/models/pt/` if present and skips otherwise. **`pytest` now defaults to a temp-file SQLite DB** (the desktop profile) and needs no services; set `DATABASE_URL=postgresql+psycopg://...annotate_test` to run against the server stack. `.github/workflows/ci.yml` runs both.

Frontend: `npm run lint` (ESLint), `npx tsc -b && npx vitest run` (typecheck + tests), `npx vitest run path/to/file.test.tsx` for a single file, `npm run build`.

### Two runtime profiles

The same backend code runs two ways, selected by `ALF_TASK_QUEUE`:

| | `celery` (docker-compose / server) | `local` (default; desktop app) |
|---|---|---|
| DB | PostgreSQL (`alembic upgrade head`) | SQLite file, schema via `app/db/init_db.py` (`create_all` + `PRAGMA user_version`) |
| Jobs | Celery worker + Beat + Redis | in-process `ThreadPoolExecutor`s + APScheduler (`app/workers/local_queue.py`, `scheduler.py`) — `celery_app` is a shim, no `celery` import |
| Live progress | Redis (`app/workers/progress_store.py` → `RedisStore`) | in-process dict (`InMemoryStore`) |
| Frontend | Vite dev server / separate | FastAPI serves the built SPA (`FRONTEND_DIST_DIR`), one origin |
| Data dirs | repo / bind mounts | `ALF_DATA_DIR` (`%APPDATA%/AutoLabelFlow`) |

Cross-dialect column types (`GUID`, `TZDateTime`, `enum_column`) live in `app/db/types.py`; Postgres DDL is byte-identical to before, so Alembic sees no diff. `alembic/env.py` refuses a non-Postgres URL.

### Desktop app (`desktop/`)

Electron shell that spawns the bundled Python backend (`local` profile) and opens a window on it; manual "Check for updates" via `electron-updater` + public GitHub Releases. `desktop/RELEASING.md` has the build/release steps; `node desktop/scripts/build.mjs` assembles the payload, `npx electron-builder --win nsis` packages it. Optional add-on packs (GPU CUDA torch, cloud SDKs) are downloaded from **Settings → Desktop app** — specs in `app/workers/tasks/packs.py`. `backend/requirements.txt` is now the shared base; `requirements-server.txt` / `requirements-desktop.txt` / `requirements-dev.txt` layer on top.

### Windows GPU note

Plain `pip install torch` on Windows resolves to a CPU-only wheel silently. Install CUDA wheels first, separately, before `requirements.txt`: `pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision` (torch/torchvision are deliberately absent from `requirements.txt` for this reason — see the comment there). If GPU training/inference silently falls back to CPU, this is almost always why.

## Architecture

### Backend layout (`backend/app/`)

```
api/v1/          REST endpoints (thin — delegate to services)
models/          SQLAlchemy ORM
schemas/         Pydantic request/response models
services/
  inference/     DetectionModel/PoseModel interfaces + Ultralytics adapter + registry
  quality/       Pluggable quality-rule framework (see below)
  annotation/    Event-log-backed annotation CRUD
  dataset/       Versioning, video-level train/val/test split, COCO/CVAT/YOLO import+export
  review/        Active-learning difficulty scoring
  training/      TrainingProvider interface: LOCAL, KAGGLE, MODAL
  storage/       ObjectStorage interface: local filesystem, MinIO
  integrations/  Kaggle, Modal, Roboflow (import/export/browse) — all optional, guarded imports
workers/         Celery app + tasks/ (inference, video, training, quality, reconcile)
core/config.py   Single source of truth for env-driven settings (see below)
```

Three provider-abstraction interfaces are the backbone of this codebase — never branch on a concrete provider outside its `registry.py`/`factory.py`:

- **`ObjectStorage`** (`services/storage/base.py`): `local` in dev, MinIO in Docker/prod. `factory.get_storage()` is the only place that decides which.
- **`TrainingProvider`** (`services/training/provider.py`): LOCAL (Celery + your GPU), KAGGLE, MODAL. An unconfigured provider (e.g. missing `KAGGLE_USERNAME`/`KAGGLE_KEY`) must degrade gracefully — `is_configured` returns `False` and the UI disables it with a reason (`GET /api/v1/training/providers`), never a crash.
- **`DetectionModel`/`PoseModel`** (`services/inference/`): wraps whatever Ultralytics YOLO weights are registered; class list is always read from the weights, never typed in.

**Quality-rule framework** (`services/quality/`): `QualityRule` (`rule_base.py`) is the interface every heuristic implements; `analyzer.py` runs whatever's registered without knowing what any individual rule checks for. Rules self-register at import time (`register_rule(...)` as a side effect) and `registry.py` auto-discovers every module under `rules/` (including `rules/packs/`) via `pkgutil.iter_modules` — adding a heuristic is "drop a file in `rules/`", not a registry edit. A rule can be class-agnostic (always active) or belong to a named, project-toggleable `pack` (e.g. `anatomical_proximity`, which needs an auxiliary pose model attached to the project via `pose_model_id`). Per-project overrides live in `Project.quality_rule_config`.

**Annotation event log** (`services/annotation/service.py`): this module is the *only* code allowed to write to the `annotations` or `annotation_events` tables — every mutation goes through it so the event log and the projection can't drift, and the `AUTO → CORRECTED` source transition rule lives in exactly one place. A `POLYGON` shape's bbox columns are always server-recomputed from `points`, never trusted from the client.

**`core/config.py`**: every runtime tunable is a `Settings` attribute read from env/`.env`, imported as `from app.core.config import settings` — don't call `os.getenv` directly elsewhere. Required config fails fast at import time. This module also sets `ULTRALYTICS_SAFE_LOAD=1` before `ultralytics` is imported anywhere in the process (mitigates arbitrary code execution from a malicious checkpoint via pickle) — every entry point (uvicorn, Celery worker, pytest) must import `app.core.config` first, directly or transitively, for this to take effect.

### Frontend layout (`frontend/src/`)

```
components/annotation/   The instrument-register annotation canvas, right panel, toolbar
pages/                    One page per sidebar entry
store/                    Zustand — local canvas interaction state only (not server state)
services/api.ts           Typed API client
config/classColors.ts     Deterministic class-index → colour (never by class name)
```

Server state goes through TanStack Query; Zustand is reserved for ephemeral local UI/canvas state.

Design system: `frontend/DESIGN.md` (Swiss International — pure black/white/`#FF3000` accent, 0px radius always, thick borders, uppercase Inter type). Read it before touching any component styling. Key rules: `accent` (`#FF3000`) is reserved exclusively for "needs attention" (suspicious-flag borders, alerts) — never decorative, never a class colour. Box colours are assigned by class *index* (`classColors.ts`), never by class *name*, since names come from whatever model is loaded. Two density registers share one token set: "editorial" (dashboards, massive numerals, generous padding) vs. "instrument" (the annotation workspace — compact, tabular-nums, minimal chrome).

## Conventions worth knowing before editing

- **Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/).** release-please (`.github/workflows/release-please.yml`) parses `main` history to bump the version and changelog: `feat:` → minor, `fix:` → patch, `feat!:` / `BREAKING CHANGE:` → minor while pre-1.0. `chore/docs/refactor/test/ci/build` don't cut a release. The version lives in `version.txt` + `desktop/package.json` (kept in sync by release-please — don't hand-edit); `desktop/RELEASING.md` has the full flow.
- When adding a new training or storage backend, implement the `TrainingProvider`/`ObjectStorage` interface and register it — don't add conditionals to callers.
- When adding a new quality heuristic, add a module under `backend/app/services/quality/rules/` (or `rules/packs/` if it's project-toggleable) that subclasses `QualityRule` and calls `register_rule(...)` — don't touch `analyzer.py` or `registry.py`.
- Auto-annotation intentionally runs at a low confidence floor (`DEFAULT_CONFIDENCE_FLOOR = 0.20`, well below the measured foot/cone false positive at 0.336) — suspicious detections are meant to be caught by quality flags and review ranking, not filtered out by a hard confidence cutoff before a human sees them. Don't raise this floor to "fix" noisy auto-annotation.
- Mid-training cancellation stops at the next epoch boundary, not mid-epoch — there's no finer hook into a single `YOLO.train()` call (see `services/training/local_provider.py`).
- Kaggle/Modal/Roboflow integrations all use guarded imports and must work with the corresponding env vars absent — verify `is_configured`/equivalent gates before assuming a code path runs.
