# AI-Assisted Annotation & Retraining Platform

A human-in-the-loop computer-vision annotation tool built around one loop:

```
existing model → auto-annotate new footage → human corrects → versioned
dataset → train a new model → register it → it becomes the new annotator
```

It loads whatever detection model you point it at and reads that model's
own classes from its weights — nothing about the taxonomy is hardcoded.
Built and validated against a real Ultralytics YOLO detector (`detect_v1.pt`)
with a documented, measured failure mode (a player's foot mistaken for a
cone), which the platform's quality-analysis layer is designed to catch —
see `frontend/DESIGN.md` and the code comments in
`backend/app/services/quality/` for the specifics.

## Stack

- **Backend**: FastAPI, Pydantic v2, SQLAlchemy 2.x, PostgreSQL, Alembic, Celery + Redis, Ultralytics YOLO
- **Frontend**: React, TypeScript, Vite, Tailwind CSS (Swiss International design system — see `frontend/DESIGN.md`)
- **Storage**: local filesystem in dev, MinIO (S3-compatible) in Docker/prod, behind one `ObjectStorage` interface
- **Training**: local (your GPU) or Kaggle, behind one `TrainingProvider` interface

## Run it

```bash
docker compose up --build
```

That's the whole thing — Postgres, Redis, MinIO, the API, the Celery
worker, and the frontend all start together. Then:

- Frontend: http://localhost:5173
- API docs: http://localhost:8000/docs
- MinIO console: http://localhost:9001 (`minioadmin` / `minioadmin`)

The backend container runs `alembic upgrade head` on startup, so the schema
is always current — no separate migration step.

### GPU training in Docker

The backend/worker images default to CUDA 13.0 wheels
(`backend/Dockerfile`'s `TORCH_INDEX_URL` build arg). For a CPU-only
deploy: `docker compose build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu backend worker`.
Local (non-Docker) GPU training needs the NVIDIA Container Toolkit and a
`worker` service configured with GPU access — see Docker Compose's
[`deploy.resources.reservations.devices`](https://docs.docker.com/compose/gpu-support/)
docs; not enabled by default since it's host-specific.

## Local development (without Docker)

Faster iteration loop — run Postgres/Redis/MinIO in Docker, everything
else directly:

```bash
docker compose up -d postgres redis minio

cd backend
python -m venv venv
./venv/Scripts/pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision  # or the cpu index — see requirements.txt
./venv/Scripts/pip install -r requirements.txt
cp .env.example .env
./venv/Scripts/python -m alembic upgrade head
./venv/Scripts/python -m uvicorn app.main:app --reload

# separately, for background jobs (video extraction, auto-annotation, training, quality analysis):
./venv/Scripts/python -m celery -A app.workers.celery_app worker -Q gpu,default -c 1 --pool=solo  # --pool=solo on Windows
```

```bash
cd frontend
npm install
npm run dev
```

> **Windows note**: plain `pip install torch` resolves to a CPU-only wheel
> here — see the comment at the top of `backend/requirements.txt`. If GPU
> training/inference silently falls back to CPU, this is why.

### Registering your first model

The app ships with no models — register one by pointing at a weights file:

```bash
curl -X POST http://localhost:8000/api/v1/models \
  -H "Content-Type: application/json" \
  -d '{"name": "detect_v1", "weights_path": "/absolute/path/to/detect_v1.pt", "kind": "DETECTOR"}'
```

or use the Models page in the UI. The class list is read from the weights
automatically. An optional auxiliary pose model (`kind: "POSE"`) can be
registered and attached to a project (`pose_model_id`) to enable the
anatomical-proximity quality-flag pack — see below.

## Tests

```bash
# backend — requires Postgres+Redis reachable at localhost (docker compose up -d postgres redis minio)
cd backend
DATABASE_URL=postgresql+psycopg://annotate:annotate@localhost:5432/annotate_test ./venv/Scripts/python -m pytest tests/ -q

# frontend
cd frontend
npx tsc -b && npx vitest run
```

Backend tests run fully offline against fake model objects (no real
weights loaded) except `test_real_model_integration.py`, which loads real
weights if present at `artifacts/models/pt/` and is skipped otherwise.

## Project layout

```
backend/app/
  api/v1/          REST endpoints
  models/          SQLAlchemy ORM
  schemas/         Pydantic request/response models
  services/
    inference/     DetectionModel/PoseModel interfaces + Ultralytics adapter
    quality/       Pluggable quality-rule framework + the anatomical_proximity pack
    annotation/    Event-log-backed annotation CRUD (source: AUTO/HUMAN/CORRECTED)
    dataset/       Versioning, video-level train/val/test split, YOLO export
    review/        Active-learning difficulty scoring
    training/      TrainingProvider interface (LOCAL, KAGGLE)
    storage/       ObjectStorage interface (local, MinIO)
  workers/         Celery app + background tasks (inference, video, training, quality)

frontend/src/
  components/annotation/   The instrument-register annotation canvas, right panel, toolbar
  pages/                   One page per sidebar entry
  store/                   Zustand — local canvas interaction state only
  services/api.ts          Typed API client
  config/classColors.ts    Deterministic class-index → colour (never by name)

frontend/DESIGN.md          The Swiss International design system reference
```

## What's deliberately not automated

- **Kaggle training** requires `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars. The
  app works fully without them — the Kaggle option is cleanly disabled
  with a reason in the UI (`GET /api/v1/training/providers`).
- **Mid-training cancellation** stops at the next epoch boundary, not
  mid-epoch — there's no finer hook into a single `YOLO.train()` call. See
  the comment in `backend/app/services/training/local_provider.py`.
