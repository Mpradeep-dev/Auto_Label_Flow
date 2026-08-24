from __future__ import annotations

from app.services.video.sampler import compute_sample_indices


def test_interval_takes_priority_over_fps() -> None:
    indices = compute_sample_indices(total_frames=100, video_fps=30.0, interval=10, fps=1.0)
    assert indices == list(range(0, 100, 10))


def test_fps_based_sampling_converts_to_a_frame_step() -> None:
    # 30fps video, sample at 5fps -> every 6th frame
    indices = compute_sample_indices(total_frames=60, video_fps=30.0, interval=None, fps=5.0)
    assert indices == list(range(0, 60, 6))


def test_defaults_to_configured_interval_when_nothing_given() -> None:
    from app.core.config import settings

    indices = compute_sample_indices(total_frames=50, video_fps=30.0)
    assert indices == list(range(0, 50, settings.DEFAULT_FRAME_SAMPLE_INTERVAL))


def test_short_clip_still_yields_several_frames_with_default_interval() -> None:
    """The concrete reason the default interval is 5, not a naive 10: this
    corpus's shortest player clips run ~54 frames — 'every 10 frames'
    would yield only 5 samples. See PLAN 'Sample footage characteristics'."""
    indices = compute_sample_indices(total_frames=54, video_fps=30.0)
    assert len(indices) >= 10


def test_zero_frames_yields_empty_list() -> None:
    assert compute_sample_indices(total_frames=0, video_fps=30.0) == []


def test_never_produces_an_out_of_range_index() -> None:
    indices = compute_sample_indices(total_frames=37, video_fps=25.0, interval=10)
    assert all(0 <= i < 37 for i in indices)
