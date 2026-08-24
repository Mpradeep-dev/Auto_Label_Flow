from __future__ import annotations

import uuid

from app.models.project import Project
from app.services.quality.registry import get_active_rules


def _project(class_config, pose_model_id=None, quality_rule_config=None) -> Project:
    return Project(
        name="p", slug="p", class_config=class_config,
        quality_rule_config=quality_rule_config or {}, pose_model_id=pose_model_id,
    )


def test_anatomical_proximity_active_by_default_when_pose_configured_and_cone_class_present() -> None:
    project = _project([{"id": 1, "name": "cone"}], pose_model_id=uuid.uuid4())
    active = get_active_rules(project)
    active_flag_types = {r.flag_type.value for r, _ in active}
    assert "CONE_NEAR_PLAYER" in active_flag_types
    assert "SUSPICIOUS_CONE" in active_flag_types


def test_anatomical_proximity_inactive_without_pose_model() -> None:
    project = _project([{"id": 1, "name": "cone"}], pose_model_id=None)
    active = get_active_rules(project)
    active_flag_types = {r.flag_type.value for r, _ in active}
    assert "CONE_NEAR_PLAYER" not in active_flag_types


def test_anatomical_proximity_inactive_when_no_cone_like_class() -> None:
    project = _project([{"id": 0, "name": "widget"}], pose_model_id=uuid.uuid4())
    active = get_active_rules(project)
    active_flag_types = {r.flag_type.value for r, _ in active}
    assert "CONE_NEAR_PLAYER" not in active_flag_types


def test_anatomical_proximity_can_be_explicitly_disabled() -> None:
    project = _project(
        [{"id": 1, "name": "cone"}],
        pose_model_id=uuid.uuid4(),
        quality_rule_config={"anatomical_proximity": {"enabled": False}},
    )
    active = get_active_rules(project)
    active_flag_types = {r.flag_type.value for r, _ in active}
    assert "CONE_NEAR_PLAYER" not in active_flag_types


def test_target_class_ids_default_to_every_class_named_cone_like() -> None:
    project = _project(
        [{"id": 0, "name": "ball"}, {"id": 1, "name": "cone"}, {"id": 2, "name": "cone_1"}],
        pose_model_id=uuid.uuid4(),
    )
    active = get_active_rules(project)
    cone_near_player = next(r for r, _ in active if r.flag_type.value == "CONE_NEAR_PLAYER")
    _, params = next((r, p) for r, p in active if r is cone_near_player)
    assert set(params["target_class_ids"]) == {1, 2}


def test_class_agnostic_rules_always_active_regardless_of_pose_config() -> None:
    project = _project([{"id": 0, "name": "widget"}], pose_model_id=None)
    active_flag_types = {r.flag_type.value for r, _ in get_active_rules(project)}
    assert "LOW_CONFIDENCE" in active_flag_types
    assert "POSSIBLE_DUPLICATE" in active_flag_types
    assert "VERY_SMALL_CONE" in active_flag_types
    assert "ISOLATED_DETECTION" in active_flag_types
    assert "TEMPORAL_ANOMALY" in active_flag_types
