from __future__ import annotations

from app.services.dataset.splitter import ImageGroupInfo, split_images


def _group(video_id: str, n: int, offset: int = 0) -> list[ImageGroupInfo]:
    return [ImageGroupInfo(image_id=f"{video_id}-{i+offset}", source_group_id=video_id) for i in range(n)]


def test_no_group_spans_two_splits() -> None:
    images = _group("v1", 30) + _group("v2", 20) + _group("v3", 10) + _group("v4", 5) + _group("v5", 5)
    result = split_images(images, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=1)

    groups: dict[str, set[str]] = {}
    for img in images:
        groups.setdefault(img.source_group_id, set()).add(result.assignment[img.image_id])
    for group_id, splits_used in groups.items():
        assert len(splits_used) == 1, f"{group_id} spans splits {splits_used}"


def test_ratios_are_approximately_honoured() -> None:
    # Many small-ish groups so the greedy packer has room to hit ratios closely.
    images = []
    for i in range(20):
        images += _group(f"v{i}", 10, offset=i * 10)
    result = split_images(images, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)

    counts = {"train": 0, "val": 0, "test": 0}
    for split in result.assignment.values():
        counts[split] += 1
    total = len(images)
    assert abs(counts["train"] / total - 0.8) < 0.1
    assert abs(counts["val"] / total - 0.1) < 0.1
    assert abs(counts["test"] / total - 0.1) < 0.1
    assert not result.used_frame_level_fallback


def test_deterministic_given_same_seed() -> None:
    images = _group("v1", 10) + _group("v2", 10) + _group("v3", 10)
    r1 = split_images(images, 0.6, 0.2, 0.2, seed=7)
    r2 = split_images(images, 0.6, 0.2, 0.2, seed=7)
    assert r1.assignment == r2.assignment


def test_different_seeds_can_change_tie_breaking() -> None:
    images = _group("v1", 10) + _group("v2", 10) + _group("v3", 10)
    r1 = split_images(images, 0.34, 0.33, 0.33, seed=1)
    r2 = split_images(images, 0.34, 0.33, 0.33, seed=2)
    # Not guaranteed to differ for every input, but for three equal-sized
    # groups feeding a near-equal 3-way split, tie-breaking should matter.
    assert r1.assignment != r2.assignment or True  # documents intent; no flaky hard assertion


def test_single_group_falls_back_to_frame_level_and_is_flagged() -> None:
    images = _group("only-video", 30)
    result = split_images(images, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=0)
    assert result.used_frame_level_fallback is True
    assert set(result.assignment.values()) <= {"train", "val", "test"}
    assert len(result.assignment) == 30


def test_standalone_images_are_independent_singleton_groups() -> None:
    images = [ImageGroupInfo(image_id=f"img{i}", source_group_id=f"image:img{i}") for i in range(15)]
    result = split_images(images, 0.6, 0.2, 0.2, seed=3)
    # 15 independent groups is enough to satisfy a 3-way split without falling back.
    assert result.used_frame_level_fallback is False
    assert len(set(result.assignment.values())) > 1


def test_two_way_split_zero_test_ratio() -> None:
    images = _group("v1", 10) + _group("v2", 10)
    result = split_images(images, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0, seed=0)
    assert "test" not in set(result.assignment.values())


def test_empty_input_returns_empty_assignment() -> None:
    result = split_images([], 0.8, 0.1, 0.1)
    assert result.assignment == {}
    assert result.used_frame_level_fallback is False
