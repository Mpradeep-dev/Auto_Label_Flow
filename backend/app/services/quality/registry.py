"""Rule registry — mirrors the sibling repo's `drills/registry.py` pattern:
each rule module calls `register_rule(...)` as an import-time side effect;
`load_all_rules()` auto-discovers every module under `rules/` (including
`rules/packs/`) via `pkgutil.iter_modules`, so adding a new heuristic is
"drop a file in `rules/`" — no registry edits.
"""
from __future__ import annotations

import importlib
import pkgutil

from app.models.project import Project
from app.services.quality.rule_base import QualityRule

_RULES: list[QualityRule] = []
_LOADED = False


def register_rule(rule: QualityRule) -> None:
    _RULES.append(rule)


def _load_package(package_name: str) -> None:
    package = importlib.import_module(package_name)
    for _, name, is_pkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if is_pkg:
            _load_package(name)
        else:
            importlib.import_module(name)


def load_all_rules() -> None:
    global _LOADED
    if _LOADED:
        return
    _load_package("app.services.quality.rules")
    _LOADED = True


def get_all_rules() -> list[QualityRule]:
    load_all_rules()
    return list(_RULES)


def get_active_rules(project: Project) -> list[tuple[QualityRule, dict]]:
    """Returns (rule, params) pairs — class-agnostic rules always included;
    pack rules included only when their pack is enabled for this project
    AND (if `requires_pose`) the project has an auxiliary pose model
    configured. `params` merges the rule's defaults with the project's
    `quality_rule_config[pack_name]` overrides."""
    active: list[tuple[QualityRule, dict]] = []
    quality_config = project.quality_rule_config or {}

    for rule in get_all_rules():
        if rule.requires_pose and not project.pose_model_id:
            continue

        if rule.pack_name is not None:
            pack_config = quality_config.get(rule.pack_name, {})
            enabled = pack_config.get("enabled", _default_pack_enabled(project, rule.pack_name))
            if not enabled:
                continue
            params = {**rule.default_params, **pack_config}
            if "target_class_ids" not in pack_config:
                params["target_class_ids"] = _default_target_class_ids(project, rule.pack_name)
        else:
            params = dict(rule.default_params)

        active.append((rule, params))

    return active


def _default_pack_enabled(project: Project, pack_name: str) -> bool:
    """A pack with no explicit project config defaults to ON if the
    project's class list plausibly matches what it targets — e.g.
    anatomical_proximity defaults on when some class name contains "cone".
    This is a heuristic default, always overridable via
    `quality_rule_config`, never a hardcoded assumption about class ids."""
    if pack_name == "anatomical_proximity":
        return bool(_default_target_class_ids(project, pack_name))
    return False


def _default_target_class_ids(project: Project, pack_name: str) -> list[int]:
    if pack_name == "anatomical_proximity":
        return [entry["id"] for entry in project.class_config if "cone" in entry.get("name", "").lower()]
    return []
