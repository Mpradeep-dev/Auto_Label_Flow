from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.slugify import slugify
from app.db.session import get_db
from app.models.image import Image
from app.models.project import Project
from app.models.video import Video
from app.schemas.project import ProjectCreate, ProjectRead, ProjectUpdate
from app.services.storage.factory import get_storage

router = APIRouter(prefix="/projects", tags=["projects"])


@router.post("", response_model=ProjectRead, status_code=status.HTTP_201_CREATED)
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)) -> Project:
    base_slug = slugify(payload.name)
    slug = base_slug
    suffix = 1
    while db.scalar(select(Project).where(Project.slug == slug)) is not None:
        suffix += 1
        slug = f"{base_slug}-{suffix}"

    project = Project(name=payload.name, slug=slug, description=payload.description)
    db.add(project)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "A project with that name already exists")
    db.refresh(project)
    return project


@router.get("", response_model=list[ProjectRead])
def list_projects(db: Session = Depends(get_db)) -> list[Project]:
    return list(db.scalars(select(Project).order_by(Project.created_at.desc())))


@router.get("/{project_id}", response_model=ProjectRead)
def get_project(project_id: uuid.UUID, db: Session = Depends(get_db)) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    return project


@router.patch("/{project_id}", response_model=ProjectRead)
def update_project(project_id: uuid.UUID, payload: ProjectUpdate, db: Session = Depends(get_db)) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")

    data = payload.model_dump(exclude_unset=True)
    for field, value in data.items():
        setattr(project, field, value)

    db.commit()
    db.refresh(project)
    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: uuid.UUID, db: Session = Depends(get_db)) -> None:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")

    # Same reasoning as datasets.py's delete_dataset (audit finding BE-02) —
    # captured before the cascade delete removes the Image/Video rows.
    image_keys = list(db.scalars(select(Image.storage_key).where(Image.project_id == project_id)))
    video_keys = list(db.scalars(select(Video.storage_key).where(Video.project_id == project_id)))

    db.delete(project)
    db.commit()

    storage = get_storage()
    for key in image_keys + video_keys:
        storage.delete(key)
