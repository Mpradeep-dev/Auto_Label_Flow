from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import (
    annotations,
    dataset_versions,
    datasets,
    health,
    images,
    inference_jobs,
    integrations,
    models,
    projects,
    quality,
    system,
    training_jobs,
    videos,
)

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(projects.router)
api_router.include_router(datasets.router)
api_router.include_router(images.router)
api_router.include_router(models.router)
api_router.include_router(annotations.router)
api_router.include_router(videos.router)
api_router.include_router(inference_jobs.router)
api_router.include_router(dataset_versions.router)
api_router.include_router(training_jobs.router)
api_router.include_router(quality.router)
api_router.include_router(integrations.router)
api_router.include_router(system.router)
