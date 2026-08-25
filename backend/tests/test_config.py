"""Regression coverage for the ARTIFACTS_DIR-under-Docker bug: `MODELS_DIR`
and `TRAINING_OUTPUT_DIR` used to be independently-computed defaults that
only agreed with `ARTIFACTS_DIR` by coincidence of the local-venv directory
layout — overriding `ARTIFACTS_DIR` (as docker-compose.yml now does, to
correct for the Dockerfile's flatter layout) silently left them pointing
at the stale, wrong default. They're computed fields now; this pins that
they actually derive from whatever `ARTIFACTS_DIR` a given instance has,
not a module-level constant computed once at import time."""
from __future__ import annotations

from pathlib import Path

from app.core.config import Settings


def test_models_dir_and_training_output_dir_derive_from_artifacts_dir() -> None:
    custom = Settings(ARTIFACTS_DIR="/custom/artifacts")
    assert custom.ARTIFACTS_DIR == Path("/custom/artifacts")
    assert custom.MODELS_DIR == Path("/custom/artifacts/models")
    assert custom.TRAINING_OUTPUT_DIR == Path("/custom/artifacts/training_runs")


def test_default_artifacts_dir_layout_is_internally_consistent() -> None:
    default = Settings()
    assert default.MODELS_DIR == default.ARTIFACTS_DIR / "models"
    assert default.TRAINING_OUTPUT_DIR == default.ARTIFACTS_DIR / "training_runs"
