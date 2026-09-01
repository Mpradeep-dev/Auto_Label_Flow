"""COCO-JSON parsing shared by every importer that reads COCO-shaped
labels: the zip importer (`import_coco.py`) and the Azure-Blob
reference-in-place importer (`integrations/azure_blob_import.py`). Kept
here so the two can't drift on how a `segmentation` / `bbox` becomes an
annotation.
"""
from __future__ import annotations

from app.models.annotation import ShapeType


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def polygon_points_from_segmentation(
    segmentation, width: int, height: int
) -> list[list[float]] | None:
    """COCO's `segmentation` is either a list of polygon rings
    (`[[x1,y1,x2,y2,...], ...]`) or an RLE dict (`{"counts": ..., "size": [h,w]}`).
    This app stores masks as polygons, so only the polygon-ring form is
    imported; RLE stays explicitly unsupported and the caller falls back to
    the bbox path. Only the first ring is used (single outer ring, no
    holes)."""
    if not isinstance(segmentation, list) or not segmentation:
        return None
    ring = segmentation[0]
    if not isinstance(ring, list) or len(ring) < 6:  # need >=3 points * 2 coords
        return None
    flat = [float(v) for v in ring]
    points = [[clamp01(flat[i] / width), clamp01(flat[i + 1] / height)] for i in range(0, len(flat), 2)]
    return points if len(points) >= 3 else None


def parse_coco(coco: dict) -> tuple[dict[int, str], dict[int, list[dict]]]:
    """`(categories_by_id, annotations_by_image_id)` from a loaded COCO dict."""
    categories = {c["id"]: c["name"] for c in coco.get("categories", [])}
    annotations_by_image: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)
    return categories, annotations_by_image


def coco_ann_to_shape_kwargs(ann: dict, width: int, height: int) -> dict:
    """Geometry-only kwargs for `annotation.service.create_annotation`:
    a `{shape_type: POLYGON, points: [...]}` pair when the annotation
    carries a usable polygon `segmentation`, otherwise a normalized
    `{x1, y1, x2, y2}` bbox from `ann["bbox"]` (COCO `[x, y, w, h]`)."""
    points = polygon_points_from_segmentation(ann.get("segmentation"), width, height)
    if points is not None:
        return {"shape_type": ShapeType.POLYGON, "points": points}
    bx, by, bw, bh = ann["bbox"]
    return {
        "x1": clamp01(bx / width),
        "y1": clamp01(by / height),
        "x2": clamp01((bx + bw) / width),
        "y2": clamp01((by + bh) / height),
    }
