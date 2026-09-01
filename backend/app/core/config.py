"""
core/config.py
===============
Centralised, environment-driven configuration, validated once at import time.

Every runtime tunable is read from the process environment (or a `.env` file
at the backend root) and exposed as an attribute on `settings`. Other modules
import `settings` from here rather than calling `os.getenv` directly, so there
is a single source of truth for defaults — mirrors the house convention in
the sibling `gsp-video-ai-processing-service` repo's `config.py`.

Required vars fail fast at import (a broken deploy should die on startup, not
three requests later inside a DB call). Optional integrations (Kaggle) are
allowed to be unset — see `KaggleSettings.is_configured`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

from pydantic import computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Registering a model loads its weights file through Ultralytics/torch, which
# is pickle-based and — by default — can execute arbitrary code embedded in a
# malicious checkpoint (see audit finding SEC-01). The vendored ultralytics
# build here has a first-class opt-in mitigation (`ULTRALYTICS_SAFE_LOAD`,
# read once at import time by `ultralytics.utils`): it restricts checkpoint
# loading to `torch.load(weights_only=True)` plus an allow-list of known
# nn.Module classes, so a file that isn't a real YOLO checkpoint fails to
# load instead of executing. This has to be set before `ultralytics` is
# imported anywhere in the process — every module that touches it imports
# `app.core.config` first (directly or transitively), so setting it here,
# at the top of the most-upstream module, is the one place that's
# guaranteed to run first regardless of entry point (uvicorn, the Celery
# worker, or pytest).
os.environ.setdefault("ULTRALYTICS_SAFE_LOAD", "1")

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
# Correct for a local venv run (config.py sits three levels under the repo
# root: backend/app/core/config.py). Under Docker this is wrong — the
# Dockerfile COPYs `backend/`'s own contents to `/app`, so there's no
# separate "project root" above it the way there is in the repo checkout,
# and this would resolve to `/artifacts` instead of the volume-mounted
# `/app/artifacts` (see docker-compose.yml's `ARTIFACTS_DIR=/app/artifacts`
# override for `backend`/`worker`, which is what actually makes this
# correct in that environment).
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOCAL_STORAGE_DIR = PROJECT_ROOT / "storage" / "local"

# Desktop app: Electron passes ALF_DATA_DIR (= app.getPath('userData'),
# e.g. %APPDATA%/AutoLabelFlow) and drops a user-editable `.env` there. Pick
# the env file before the Settings class body is evaluated.
_ALF_DATA_DIR_ENV = os.environ.get("ALF_DATA_DIR")
_ENV_FILE = (
    Path(_ALF_DATA_DIR_ENV).expanduser() / ".env" if _ALF_DATA_DIR_ENV else BACKEND_DIR / ".env"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    APP_NAME: str = "AI-Assisted Annotation & Retraining Platform"
    ENV: Literal["development", "production", "test"] = "development"
    DEBUG: bool = True

    # --- Desktop packaging ---
    # Set by the Electron shell. When present, all mutable state (SQLite DB,
    # uploaded media, model artifacts, logs) is rooted here instead of under
    # the (read-only, when frozen) install tree, and the defaults below flip
    # to the standalone profile: SQLite, local storage, no debug.
    ALF_DATA_DIR: str | None = None
    # Job execution backend. `local` = in-process ThreadPoolExecutor +
    # APScheduler (desktop). `celery` = Celery + Redis (docker-compose/server).
    ALF_TASK_QUEUE: Literal["local", "celery"] = "local"
    # Version string surfaced by /api/v1/health and /api/v1/system/info.
    # Electron injects ALF_APP_VERSION (= app.getVersion()); otherwise a
    # build-time-generated app/_version.py; otherwise a dev sentinel.
    ALF_APP_VERSION: str | None = None
    # When set, FastAPI also serves the built SPA from this directory (the
    # desktop app is one origin). Unset in dev/server — the Vite dev server
    # serves the frontend there.
    FRONTEND_DIST_DIR: Path | None = None
    LOG_DIR: Path | None = None

    # --- CORS ---
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # --- Database (required — fail fast if unreachable-by-config) ---
    DATABASE_URL: str = "postgresql+psycopg://annotate:annotate@localhost:5432/annotate"

    # --- Redis / Celery ---
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str | None = None
    CELERY_RESULT_BACKEND: str | None = None

    @computed_field  # type: ignore[misc]
    @property
    def celery_broker_url(self) -> str:
        return self.CELERY_BROKER_URL or self.REDIS_URL

    @computed_field  # type: ignore[misc]
    @property
    def celery_result_backend(self) -> str:
        return self.CELERY_RESULT_BACKEND or self.REDIS_URL

    # --- Storage ---
    STORAGE_BACKEND: Literal["local", "minio", "azure"] = "local"
    LOCAL_STORAGE_DIR: Path = LOCAL_STORAGE_DIR

    MINIO_ENDPOINT: str = "localhost:9000"
    # Host used only when signing presigned GET URLs handed to the browser.
    # Inside Docker Compose, MINIO_ENDPOINT is the internal service name
    # ("minio:9000") so the backend/worker containers can reach it, but a
    # browser on the host can't resolve that name — presigned URLs must be
    # signed against the host-reachable address instead. Defaults to
    # MINIO_ENDPOINT so a non-Docker local setup (where both already point
    # at localhost) needs no extra config.
    MINIO_PUBLIC_ENDPOINT: str | None = None
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "annotate"
    MINIO_SECURE: bool = False
    # Passed explicitly to both Minio clients so presigned_get_object never
    # triggers its own network round-trip (GetBucketLocation) against
    # MINIO_ENDPOINT to discover it — that call would use the *public*
    # client's endpoint, which is unreachable from inside the container.
    # "us-east-1" matches the server default a bucket gets when created
    # (as ours is, in MinioStorage.__init__) without an explicit location.
    MINIO_REGION: str = "us-east-1"

    @computed_field  # type: ignore[misc]
    @property
    def minio_public_endpoint(self) -> str:
        return self.MINIO_PUBLIC_ENDPOINT or self.MINIO_ENDPOINT

    # --- Azure Blob (optional; only read when STORAGE_BACKEND=azure) ---
    AZURE_STORAGE_CONNECTION_STRING: str | None = None
    AZURE_STORAGE_CONTAINER: str = "annotate"

    # --- Uploads ---
    MAX_UPLOAD_SIZE_MB: int = 200
    ALLOWED_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
    ALLOWED_VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv")

    # --- Models ---
    ARTIFACTS_DIR: Path = ARTIFACTS_DIR

    # Derived from ARTIFACTS_DIR rather than a second independently
    # env-overridable default — the bug this fixes was exactly that kind of
    # drift: overriding ARTIFACTS_DIR alone used to leave this pointing at
    # a stale, separately-computed default that disagreed with it.
    @computed_field  # type: ignore[misc]
    @property
    def MODELS_DIR(self) -> Path:
        return self.ARTIFACTS_DIR / "models"

    # --- Inference defaults ---
    # Auto-annotation runs at a low confidence floor by design: on the
    # measured ex22_Mayur.mov clip the documented foot-false-positive cone
    # scored 0.336, below production's detect_conf=0.55 stopgap. A platform
    # meant to catch that failure mode must not filter it out before a human
    # ever sees it — filtering happens via quality flags and review ranking,
    # not a hard confidence cutoff. See PLAN "Measured numbers" section.
    DEFAULT_CONFIDENCE_FLOOR: float = 0.20
    DEFAULT_IOU_THRESHOLD: float = 0.70
    DEFAULT_INFERENCE_IMGSZ: int = 640
    DEFAULT_INFERENCE_BATCH_SIZE: int = 8

    # --- Video sampling ---
    # Sample footage in this corpus runs 1.8-7s at ~30fps (54-203 frames) —
    # short clips. "Every 10 frames" (a common default elsewhere) would
    # yield as few as 5 frames from the shortest clips, so the default here
    # is denser.
    DEFAULT_FRAME_SAMPLE_INTERVAL: int = 5

    # --- Kaggle (optional; unset means the provider registers as disabled) ---
    KAGGLE_USERNAME: str | None = None
    KAGGLE_KEY: str | None = None

    @computed_field  # type: ignore[misc]
    @property
    def kaggle_configured(self) -> bool:
        return bool(self.KAGGLE_USERNAME and self.KAGGLE_KEY)

    # --- Modal (optional; unset means the provider registers as disabled) ---
    MODAL_TOKEN_ID: str | None = None
    MODAL_TOKEN_SECRET: str | None = None

    @computed_field  # type: ignore[misc]
    @property
    def modal_configured(self) -> bool:
        return bool(self.MODAL_TOKEN_ID and self.MODAL_TOKEN_SECRET)

    # --- GPU / training ---
    TRAINING_DEVICE: str = "0"  # torch device string; "cpu" falls back cleanly

    @computed_field  # type: ignore[misc]
    @property
    def TRAINING_OUTPUT_DIR(self) -> Path:
        return self.ARTIFACTS_DIR / "training_runs"

    @computed_field  # type: ignore[misc]
    @property
    def APP_VERSION(self) -> str:
        if self.ALF_APP_VERSION:
            return self.ALF_APP_VERSION
        try:
            from app._version import __version__  # generated at build time

            return __version__
        except Exception:
            return "0.0.0-dev"

    @model_validator(mode="after")
    def _apply_desktop_profile(self) -> "Settings":
        """When ALF_DATA_DIR is set (packaged desktop app), root all mutable
        state there and flip unset defaults to the standalone profile. An
        explicitly-provided value (env var / .env) always wins."""
        if not self.ALF_DATA_DIR:
            return self
        data = Path(self.ALF_DATA_DIR).expanduser().resolve()
        given = self.model_fields_set
        if "DATABASE_URL" not in given:
            self.DATABASE_URL = f"sqlite+pysqlite:///{(data / 'data' / 'app.db').as_posix()}"
        if "LOCAL_STORAGE_DIR" not in given:
            self.LOCAL_STORAGE_DIR = data / "storage"
        if "ARTIFACTS_DIR" not in given:
            self.ARTIFACTS_DIR = data / "artifacts"
        if "LOG_DIR" not in given:
            self.LOG_DIR = data / "logs"
        if "STORAGE_BACKEND" not in given:
            self.STORAGE_BACKEND = "local"
        if "DEBUG" not in given:
            self.DEBUG = False
        if "ENV" not in given:
            self.ENV = "production"
        return self


settings = Settings()

# Fail fast on structurally invalid config (not "is the DB up", which is a
# runtime concern handled by the health endpoint — but a config value that's
# nonsensical on its face should die at import, not after accepting traffic).
if settings.MAX_UPLOAD_SIZE_MB <= 0:
    raise ValueError("MAX_UPLOAD_SIZE_MB must be positive")
if not (0.0 <= settings.DEFAULT_CONFIDENCE_FLOOR <= 1.0):
    raise ValueError("DEFAULT_CONFIDENCE_FLOOR must be in [0, 1]")
if settings.STORAGE_BACKEND == "azure" and not settings.AZURE_STORAGE_CONNECTION_STRING:
    raise ValueError("AZURE_STORAGE_CONNECTION_STRING is required when STORAGE_BACKEND=azure")

# A frozen build with no data dir would silently mkdir under Program Files /
# the CWD and "work" until the first write permission error mid-session.
if getattr(sys, "frozen", False) and not settings.ALF_DATA_DIR:
    raise RuntimeError("ALF_DATA_DIR must be set when running as a packaged (frozen) app")

_dirs_to_make = [settings.ARTIFACTS_DIR, settings.MODELS_DIR / "pt", settings.LOCAL_STORAGE_DIR]
if settings.LOG_DIR is not None:
    _dirs_to_make.append(settings.LOG_DIR)
if settings.DATABASE_URL.startswith("sqlite"):
    _db_path = settings.DATABASE_URL.split(":///", 1)[-1]
    if _db_path:
        _dirs_to_make.append(Path(_db_path).parent)
for _dir in _dirs_to_make:
    _dir.mkdir(parents=True, exist_ok=True)
