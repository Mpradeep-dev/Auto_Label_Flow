"""Adapted from gsp-video-ai-processing-service
`fcg-ai-video-processing/analytics/tracking/temporal.py::TemporalAnalyzer.body_scale`.

This is the unit the anatomical_proximity thresholds (0.75 BL, 1.538 BL —
see PLAN "Measured numbers") are expressed in: hip-to-ankle leg length,
falling back to torso length, then a constant. The sibling repo's version
averages over a list of video frames; this platform annotates single
images just as often as video, so it's adapted to accept ONE set of COCO
keypoints (a single PoseResult) and returns which fallback tier fired, so
callers can record `body_scale_source` on the flag (PLAN: "so a reviewer
can see when a call rested on the constant rather than measured anatomy").
"""
from __future__ import annotations

from app.services.inference.detector import Keypoint
from app.services.quality.geometry import distance

_LEFT_HIP, _RIGHT_HIP = 11, 12
_LEFT_ANKLE, _RIGHT_ANKLE = 15, 16
_LEFT_SHOULDER, _RIGHT_SHOULDER = 5, 6

_FALLBACK_CONSTANT = 0.4


def _midpoint(a: Keypoint, b: Keypoint) -> Keypoint:
    return Keypoint(x=(a.x + b.x) / 2, y=(a.y + b.y) / 2, confidence=min(a.confidence, b.confidence))


def compute_body_scale(
    keypoints: list[Keypoint], vis_threshold: float = 0.5, aspect: float = 1.0
) -> tuple[float, str]:
    """Returns (scale, source) where source is "leg", "torso", or "constant".

    `keypoints` must be COCO-ordered (index 0-16, as returned by
    UltralyticsPoseModel). Prefers average leg length (hip->ankle); falls
    back to torso length (shoulder midpoint -> hip midpoint); then a 0.4
    constant. Always returns a value > 0.
    """
    if len(keypoints) < 17:
        return _FALLBACK_CONSTANT, "constant"

    leg_lengths = []
    for hip_idx, ankle_idx in ((_LEFT_HIP, _LEFT_ANKLE), (_RIGHT_HIP, _RIGHT_ANKLE)):
        hip, ankle = keypoints[hip_idx], keypoints[ankle_idx]
        if hip.confidence >= vis_threshold and ankle.confidence >= vis_threshold:
            leg_lengths.append(distance(hip, ankle, aspect))
    if leg_lengths:
        return max(1e-3, sum(leg_lengths) / len(leg_lengths)), "leg"

    l_shoulder, r_shoulder = keypoints[_LEFT_SHOULDER], keypoints[_RIGHT_SHOULDER]
    l_hip, r_hip = keypoints[_LEFT_HIP], keypoints[_RIGHT_HIP]
    if (
        l_shoulder.confidence >= vis_threshold
        and r_shoulder.confidence >= vis_threshold
        and l_hip.confidence >= vis_threshold
        and r_hip.confidence >= vis_threshold
    ):
        shoulder_mid = _midpoint(l_shoulder, r_shoulder)
        hip_mid = _midpoint(l_hip, r_hip)
        return max(1e-3, distance(shoulder_mid, hip_mid, aspect)), "torso"

    return _FALLBACK_CONSTANT, "constant"
