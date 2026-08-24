"""Unit tests for the annotation service — the only code allowed to touch
`annotations`/`annotation_events` (PLAN Core design decision 2). These
exercise the invariant the acceptance-rate metric depends on: AUTO stays
AUTO on approve-only, and moves to CORRECTED the instant geometry or class
changes."""
from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.annotation import (
    Annotation,
    AnnotationEvent,
    AnnotationEventAction,
    AnnotationReviewStatus,
    AnnotationSource,
    ErrorCategory,
    ErrorReason,
)
from app.models.image import Image, ImageReviewStatus
from app.services.annotation import service as annotation_service


@pytest.fixture()
def image(db_session: Session) -> Image:
    from app.models.dataset import Dataset
    from app.models.project import Project

    project = Project(name=f"p-{uuid.uuid4().hex[:8]}", slug=f"p-{uuid.uuid4().hex[:8]}")
    db_session.add(project)
    db_session.flush()
    dataset = Dataset(project_id=project.id, name="d")
    db_session.add(dataset)
    db_session.flush()
    img = Image(
        project_id=project.id,
        dataset_id=dataset.id,
        storage_key="x/y.jpg",
        original_filename="y.jpg",
        width=100,
        height=100,
    )
    db_session.add(img)
    db_session.commit()
    db_session.refresh(img)
    return img


def test_create_human_annotation(db_session: Session, image: Image) -> None:
    ann = annotation_service.create_annotation(
        db_session,
        image_id=image.id,
        class_id=1,
        class_name="cone",
        x1=0.1,
        y1=0.1,
        x2=0.2,
        y2=0.2,
        confidence=None,
        source=AnnotationSource.HUMAN,
    )
    assert ann.source == AnnotationSource.HUMAN
    assert ann.revision_seq == 1

    events = list(db_session.scalars(select(AnnotationEvent).where(AnnotationEvent.annotation_id == ann.id)))
    assert len(events) == 1
    assert events[0].action == AnnotationEventAction.CREATE


def test_geometry_edit_moves_auto_to_corrected(db_session: Session, image: Image) -> None:
    ann = annotation_service.create_annotation(
        db_session,
        image_id=image.id,
        class_id=1,
        class_name="cone",
        x1=0.1,
        y1=0.1,
        x2=0.2,
        y2=0.2,
        confidence=0.66,
        source=AnnotationSource.AUTO,
    )
    assert ann.source == AnnotationSource.AUTO

    moved = annotation_service.update_annotation(db_session, annotation_id=ann.id, x1=0.15)
    assert moved.source == AnnotationSource.CORRECTED
    assert moved.revision_seq == 2


def test_class_change_moves_auto_to_corrected(db_session: Session, image: Image) -> None:
    ann = annotation_service.create_annotation(
        db_session,
        image_id=image.id,
        class_id=1,
        class_name="cone",
        x1=0.1,
        y1=0.1,
        x2=0.2,
        y2=0.2,
        confidence=0.66,
        source=AnnotationSource.AUTO,
    )
    moved = annotation_service.update_annotation(
        db_session, annotation_id=ann.id, class_id=0, class_name="ball"
    )
    assert moved.source == AnnotationSource.CORRECTED
    assert moved.class_name == "ball"


def test_human_annotation_stays_human_on_edit(db_session: Session, image: Image) -> None:
    ann = annotation_service.create_annotation(
        db_session,
        image_id=image.id,
        class_id=1,
        class_name="cone",
        x1=0.1,
        y1=0.1,
        x2=0.2,
        y2=0.2,
        confidence=None,
        source=AnnotationSource.HUMAN,
    )
    moved = annotation_service.update_annotation(db_session, annotation_id=ann.id, x1=0.15)
    assert moved.source == AnnotationSource.HUMAN


def test_approve_without_edit_leaves_auto_intact(db_session: Session, image: Image) -> None:
    """The core acceptance-rate invariant: approving an untouched AUTO
    prediction must NOT relabel it CORRECTED."""
    annotation_service.create_annotation(
        db_session,
        image_id=image.id,
        class_id=1,
        class_name="cone",
        x1=0.1,
        y1=0.1,
        x2=0.2,
        y2=0.2,
        confidence=0.7,
        source=AnnotationSource.AUTO,
    )
    updated_image = annotation_service.approve_image(db_session, image_id=image.id)
    assert updated_image.review_status == ImageReviewStatus.APPROVED

    [ann] = annotation_service.list_annotations_for_image(db_session, image.id)
    assert ann.source == AnnotationSource.AUTO  # unchanged
    assert ann.review_status == AnnotationReviewStatus.APPROVED


def test_reject_image_sets_status(db_session: Session, image: Image) -> None:
    updated = annotation_service.reject_image(db_session, image_id=image.id)
    assert updated.review_status == ImageReviewStatus.REJECTED


def test_delete_removes_from_projection_but_keeps_event_history(db_session: Session, image: Image) -> None:
    ann = annotation_service.create_annotation(
        db_session,
        image_id=image.id,
        class_id=1,
        class_name="cone",
        x1=0.1,
        y1=0.1,
        x2=0.2,
        y2=0.2,
        confidence=0.336,  # the measured foot-FP confidence
        source=AnnotationSource.AUTO,
    )
    annotation_id = ann.id

    annotation_service.delete_annotation(
        db_session,
        annotation_id=annotation_id,
        error_category=ErrorCategory.FALSE_POSITIVE,
        error_reason=ErrorReason.PLAYER_FOOT,
    )

    assert db_session.get(Annotation, annotation_id) is None

    events = list(
        db_session.scalars(
            select(AnnotationEvent)
            .where(AnnotationEvent.annotation_id == annotation_id)
            .order_by(AnnotationEvent.revision_seq)
        )
    )
    assert [e.action for e in events] == [AnnotationEventAction.CREATE, AnnotationEventAction.DELETE]
    assert events[-1].error_category == ErrorCategory.FALSE_POSITIVE
    assert events[-1].error_reason == ErrorReason.PLAYER_FOOT


def test_delete_missing_annotation_raises(db_session: Session) -> None:
    with pytest.raises(annotation_service.AnnotationNotFoundError):
        annotation_service.delete_annotation(db_session, annotation_id=uuid.uuid4())


def test_duplicate_creates_offset_human_copy(db_session: Session, image: Image) -> None:
    original = annotation_service.create_annotation(
        db_session,
        image_id=image.id,
        class_id=1,
        class_name="cone",
        x1=0.1,
        y1=0.1,
        x2=0.2,
        y2=0.2,
        confidence=0.7,
        source=AnnotationSource.AUTO,
    )
    copy = annotation_service.duplicate_annotation(db_session, annotation_id=original.id)
    assert copy.id != original.id
    assert copy.source == AnnotationSource.HUMAN
    assert copy.x1 > original.x1
    assert copy.confidence is None

    all_annotations = annotation_service.list_annotations_for_image(db_session, image.id)
    assert len(all_annotations) == 2
