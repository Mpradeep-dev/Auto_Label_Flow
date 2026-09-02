"""Status + on-disk layout for the two optional SAM checkpoints offered from
Settings ("SAM Lite" = MobileSAM, "SAM Full" = a larger SAM2 checkpoint).

Deliberately NOT gated on `ALF_DATA_DIR` the way `services/system/packs.py`
is — SAM-assisted segmentation is available on the server/Docker profile
too, not just the desktop app, so weights live under the same `MODELS_DIR`
model registration already uses on both profiles. This module owns only
status + paths; the download job lives in `workers/tasks/sam_download.py`,
same split as `packs.py`/`workers/tasks/packs.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings


@dataclass(frozen=True)
class SamVariant:
    name: str
    filename: str
    url: str
    label: str
    blurb: str


# Exact release-asset URLs/filenames to confirm against Ultralytics' current
# release at rollout — SAM/SAM2 checkpoint asset names have shifted across
# ultralytics versions; kept as one place to update rather than scattered.
SAM_VARIANTS: dict[str, SamVariant] = {
    "sam-lite": SamVariant(
        name="sam-lite",
        filename="mobile_sam.pt",
        url="https://github.com/ultralytics/assets/releases/download/v8.3.0/mobile_sam.pt",
        label="SAM Lite (MobileSAM)",
        blurb="~40 MB, usable on CPU. Lower mask quality than SAM Full, but works on any machine.",
    ),
    "sam-full": SamVariant(
        name="sam-full",
        filename="sam2_b.pt",
        url="https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2_b.pt",
        label="SAM Full (SAM2)",
        blurb="~150 MB+, best mask quality. GPU recommended — slow on CPU-only machines.",
    ),
}


def _sam_dir() -> Path:
    d = settings.MODELS_DIR / "sam"
    d.mkdir(parents=True, exist_ok=True)
    return d


def weights_path(variant_name: str) -> Path:
    return _sam_dir() / SAM_VARIANTS[variant_name].filename


@dataclass
class SamModelStatus:
    name: str
    label: str
    blurb: str
    installed: bool
    size_bytes: int | None


def status(variant_name: str) -> SamModelStatus:
    variant = SAM_VARIANTS[variant_name]
    path = weights_path(variant_name)
    installed = path.is_file()
    return SamModelStatus(
        name=variant.name,
        label=variant.label,
        blurb=variant.blurb,
        installed=installed,
        size_bytes=path.stat().st_size if installed else None,
    )


def all_status() -> list[SamModelStatus]:
    return [status(name) for name in SAM_VARIANTS]


def is_installed(variant_name: str) -> bool:
    return weights_path(variant_name).is_file()


def any_installed() -> bool:
    return any(is_installed(name) for name in SAM_VARIANTS)


def remove(variant_name: str) -> None:
    from app.services.inference.sam_adapter import invalidate_segmenter

    path = weights_path(variant_name)
    invalidate_segmenter(str(path))
    path.unlink(missing_ok=True)
