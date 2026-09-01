"""Optional add-on packs for the desktop app.

The base installer ships CPU-only inference and no cloud SDKs. Two packs are
downloaded on demand from a Settings button:

  - ``gpu``          — CUDA torch + torchvision, for local GPU YOLO training
  - ``integrations`` — kaggle / modal / roboflow / boto3 / azure SDKs

A pack is a zip of wheels published as a GitHub Release asset. Installing it
unpacks the wheels' ``site-packages`` under
``{ALF_DATA_DIR}/packs/<name>/site-packages`` and writes a ``pack.json``
marker; the backend prepends installed pack paths to ``sys.path`` at startup
(and to the training subprocess ``PYTHONPATH``). This module owns only the
on-disk layout + status; the download/unpack job lives in
``workers/tasks``.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings

PACK_NAMES = ("gpu", "integrations")


@dataclass
class PackStatus:
    name: str
    installed: bool
    version: str | None
    size_bytes: int | None


def _packs_root() -> Path | None:
    if not settings.ALF_DATA_DIR:
        return None
    return Path(settings.ALF_DATA_DIR).expanduser() / "packs"


def pack_dir(name: str) -> Path | None:
    root = _packs_root()
    return None if root is None else root / name


def _site_packages(name: str) -> Path | None:
    d = pack_dir(name)
    return None if d is None else d / "site-packages"


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def status(name: str) -> PackStatus:
    d = pack_dir(name)
    marker = None if d is None else d / "pack.json"
    if d is None or not marker.is_file():
        return PackStatus(name=name, installed=False, version=None, size_bytes=None)
    try:
        meta = json.loads(marker.read_text())
    except Exception:  # noqa: BLE001
        meta = {}
    sp = _site_packages(name)
    return PackStatus(
        name=name,
        installed=True,
        version=meta.get("version"),
        size_bytes=_dir_size(sp) if sp and sp.is_dir() else None,
    )


def all_status() -> list[PackStatus]:
    return [status(n) for n in PACK_NAMES]


def is_installed(name: str) -> bool:
    return status(name).installed


def activate_installed_packs() -> None:
    """Prepend every installed pack's site-packages to ``sys.path``. Called
    once at startup, before torch / the cloud SDKs might be imported."""
    for name in PACK_NAMES:
        sp = _site_packages(name)
        if sp and sp.is_dir():
            p = str(sp)
            if p not in sys.path:
                sys.path.insert(0, p)
