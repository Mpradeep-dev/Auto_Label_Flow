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

from pathlib import Path
from typing import Literal

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
LOCAL_STORAGE_DIR = PROJECT_ROOT / "storage" / "local"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    APP_NAME: str = "AI-Assisted Annotation & Retraining Platform"
    ENV: Literal["development", "production", "test"] = "development"
    DEBUG: bool = True

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
    STORAGE_BACKEND: Literal["local", "minio"] = "local"
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

    # --- Uploads ---
    MAX_UPLOAD_SIZE_MB: int = 200
    ALLOWED_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
    ALLOWED_VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv")

    # --- Models ---
    ARTIFACTS_DIR: Path = ARTIFACTS_DIR
    MODELS_DIR: Path = MODELS_DIR

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

    # --- GPU / training ---
    TRAINING_DEVICE: str = "0"  # torch device string; "cpu" falls back cleanly
    TRAINING_OUTPUT_DIR: Path = ARTIFACTS_DIR / "training_runs"


settings = Settings()

# Fail fast on structurally invalid config (not "is the DB up", which is a
# runtime concern handled by the health endpoint — but a config value that's
# nonsensical on its face should die at import, not after accepting traffic).
if settings.MAX_UPLOAD_SIZE_MB <= 0:
    raise ValueError("MAX_UPLOAD_SIZE_MB must be positive")
if not (0.0 <= settings.DEFAULT_CONFIDENCE_FLOOR <= 1.0):
    raise ValueError("DEFAULT_CONFIDENCE_FLOOR must be in [0, 1]")

for _dir in (settings.ARTIFACTS_DIR, settings.MODELS_DIR / "pt", settings.LOCAL_STORAGE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
