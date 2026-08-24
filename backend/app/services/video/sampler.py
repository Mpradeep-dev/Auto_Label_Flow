"""Frame sampling: which frame indices to extract from a video, given
either a fixed interval or a target FPS. Deliberately NOT "every frame" —
consecutive frames of a 30fps clip are near-identical, and the sample
footage in this corpus runs short (1.8-7s / 54-203 frames at ~30fps), so a
naive "every 10 frames" default would yield as few as 5 frames from the
shortest clips. `DEFAULT_FRAME_SAMPLE_INTERVAL=5` is denser for that reason
(PLAN "Sample footage characteristics").

A single pure function rather than a class hierarchy for now — this is the
seam a future motion/scene-change sampler would plug into (PLAN "Pluggable
FrameSampler registry... extensible to motion/scene-change sampling
later"), but nothing beyond fixed-interval/fixed-fps is needed yet, so a
registry would be premature abstraction.
"""
from __future__ import annotations

from app.core.config import settings


def compute_sample_indices(
    total_frames: int,
    video_fps: float,
    interval: int | None = None,
    fps: float | None = None,
) -> list[int]:
    """`interval` (every Nth frame) takes priority over `fps` if both are
    given (matches FrameSampleConfig's documented precedence). Falls back
    to the configured default interval if neither is given."""
    if total_frames <= 0:
        return []

    if interval and interval > 0:
        step = interval
    elif fps and fps > 0 and video_fps > 0:
        step = max(1, round(video_fps / fps))
    else:
        step = settings.DEFAULT_FRAME_SAMPLE_INTERVAL

    return list(range(0, total_frames, step))
