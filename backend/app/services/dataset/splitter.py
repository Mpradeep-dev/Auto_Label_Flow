"""Video-level train/val/test splitting (PLAN "YOLO export splits by video,
never by frame"). Consecutive frames of a 30fps clip are near-identical —
a per-frame random split leaks a val frame's near-duplicate neighbour into
train, inflating validation metrics artificially.

Grouping: every image with the same `source_group_id` stays in exactly one
split. A video frame's group is its `video_id`; a standalone-uploaded image
is its own singleton group — deliberately NOT "one group per upload batch"
(the plan's original wording), because two unrelated photos uploaded
together share no visual correlation the way video frames do. Grouping
them would only shrink the effective split granularity for no leakage
benefit.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class ImageGroupInfo:
    image_id: str
    source_group_id: str


@dataclass(frozen=True)
class SplitResult:
    assignment: dict[str, str]  # image_id -> "train" | "val" | "test"
    used_frame_level_fallback: bool


_SPLITS = ("train", "val", "test")


def split_images(
    images: list[ImageGroupInfo],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 0,
) -> SplitResult:
    """Greedy largest-group-first bin packing: sort groups by size
    descending (shuffled first, with `seed`, to break ties deterministically
    and reproducibly), then assign each group whole to whichever split is
    currently furthest below its target ratio. Converges much closer to the
    requested ratios than assigning groups in arbitrary order, since a
    handful of large source videos otherwise dominate total frame count.
    """
    if not images:
        return SplitResult(assignment={}, used_frame_level_fallback=False)

    total = train_ratio + val_ratio + test_ratio
    targets = {
        "train": train_ratio / total,
        "val": val_ratio / total,
        "test": test_ratio / total,
    }

    groups: dict[str, list[str]] = defaultdict(list)
    for img in images:
        groups[img.source_group_id].append(img.image_id)

    # Escape hatch: too few groups to meaningfully honour a 3-way split by
    # group (e.g. one video, or all standalone images from one shoot) —
    # fall back to a random per-image split rather than dumping everything
    # into a single split. This reintroduces the near-duplicate-frame
    # leakage the group-level algorithm exists to avoid, so it's flagged
    # on the returned result and surfaced to the user, not silent.
    non_zero_targets = sum(1 for r in (train_ratio, val_ratio, test_ratio) if r > 0)
    if len(groups) < non_zero_targets:
        rng = random.Random(seed)
        shuffled = [img.image_id for img in images]
        rng.shuffle(shuffled)
        assignment = _assign_individually(shuffled, targets)
        return SplitResult(assignment=assignment, used_frame_level_fallback=True)

    rng = random.Random(seed)
    group_ids = list(groups.keys())
    rng.shuffle(group_ids)
    group_ids.sort(key=lambda g: len(groups[g]), reverse=True)

    total_count = len(images)
    counts = {s: 0 for s in _SPLITS}
    assignment: dict[str, str] = {}

    for group_id in group_ids:
        member_ids = groups[group_id]
        # Deficit = how far below target this split currently is; assign
        # the whole group to whichever split needs it most.
        best_split = max(
            _SPLITS,
            key=lambda s: (targets[s] * total_count) - counts[s] if targets[s] > 0 else float("-inf"),
        )
        for image_id in member_ids:
            assignment[image_id] = best_split
        counts[best_split] += len(member_ids)

    return SplitResult(assignment=assignment, used_frame_level_fallback=False)


def _assign_individually(image_ids: list[str], targets: dict[str, float]) -> dict[str, str]:
    total = len(image_ids)
    counts = {s: 0 for s in _SPLITS}
    assignment: dict[str, str] = {}
    for image_id in image_ids:
        best_split = max(
            _SPLITS,
            key=lambda s: (targets[s] * total) - counts[s] if targets[s] > 0 else float("-inf"),
        )
        assignment[image_id] = best_split
        counts[best_split] += 1
    return assignment
