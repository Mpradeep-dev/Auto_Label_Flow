"""Adapted from gsp-video-ai-processing-service
`fcg-ai-video-processing/analytics/tracking/sort_tracker.py` — a class-aware
SORT/ByteTrack-style multi-object tracker. Feeds the ISOLATED_DETECTION and
TEMPORAL_ANOMALY quality rules (PLAN Phase 7): a track's `hits`/`missed`
counters and Kalman velocity are exactly the signals those rules check.

The one real change from the source: the original hardcodes `CLASS_BALL`/
`CLASS_CONE` as fixed global ids (0/1) from that repo's fixed taxonomy.
This platform's classes are read from whatever model is loaded (PLAN
Decision 1) — there is no guarantee class 0 is a ball for every project.
So the "static object, long occlusion tolerance" and "fast object, single
display" behaviors are parameterized at construction (`static_class_ids`,
`fast_class_ids`) instead of hardcoded. Any class id not in either set
falls back to the original's own default constants (`.get(id, default)`),
which is the same fallback the source already had for unknown classes —
this is a generalization of the existing pattern, not a new one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# Fallback defaults for any class not explicitly marked static or fast —
# matches the source's own `.get(track.class_id, 4)` / `.get(..., 3)` fallback.
_DEFAULT_PREDICT_MAX_MISSED = 4
_DEFAULT_PREDICT_MIN_HITS = 3
_STATIC_PREDICT_MAX_MISSED = 60  # cone-like: static once confirmed, tolerates long occlusion
_STATIC_PREDICT_MIN_HITS = 5
_FAST_PREDICT_MAX_MISSED = 12  # ball-like: short occlusion window, avoid stale "ghost" detections
_FAST_PREDICT_MIN_HITS = 3

_CENTER_DIST_GATE = 0.08
_CENTER_DIST_GATE_FAST = 0.18  # wider gate for fast-moving objects
_FAST_HIGH_CONF_CAP = 0.20


def _bbox_to_state(bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _state_to_bbox(state: np.ndarray) -> np.ndarray:
    cx, cy, w, h = [float(v) for v in state[:4]]
    w = max(1e-6, w)
    h = max(1e-6, h)
    return np.array([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dtype=np.float32)


def _bbox(det: Dict) -> np.ndarray:
    return np.array([det["x1"], det["y1"], det["x2"], det["y2"]], dtype=np.float32)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / (area_a + area_b - inter + 1e-6)


_F = np.eye(8, dtype=np.float32)
_F[0, 4] = 1.0
_F[1, 5] = 1.0
_F[2, 6] = 1.0
_F[3, 7] = 1.0

_H = np.zeros((4, 8), dtype=np.float32)
_H[0, 0] = 1.0
_H[1, 1] = 1.0
_H[2, 2] = 1.0
_H[3, 3] = 1.0

_I = np.eye(8, dtype=np.float32)
_Q = np.diag([2e-3, 2e-3, 1e-3, 1e-3, 1e-2, 1e-2, 6e-3, 6e-3]).astype(np.float32)
_Q_FAST = np.diag([8e-3, 8e-3, 2e-3, 2e-3, 4e-2, 4e-2, 1e-2, 1e-2]).astype(np.float32)
_R = np.diag([2e-2, 2e-2, 5e-2, 5e-2]).astype(np.float32)
_R_FAST = np.diag([8e-3, 8e-3, 3e-2, 3e-2]).astype(np.float32)
_Q_STATIC = np.diag([5e-4, 5e-4, 5e-4, 5e-4, 1e-6, 1e-6, 1e-6, 1e-6]).astype(np.float32)


@dataclass
class _Track:
    track_id: int
    class_id: int
    state: np.ndarray
    covariance: np.ndarray
    missed: int = 0
    hits: int = 1

    def predict(self, is_static: bool, is_fast: bool) -> np.ndarray:
        if is_static:
            q = _Q_STATIC
        elif is_fast:
            q = _Q_FAST
        else:
            q = _Q
        if is_static and self.hits >= _STATIC_PREDICT_MIN_HITS:
            self.state[4:] = 0.0  # cones don't move — zero velocity once confirmed
        self.state = _F @ self.state
        self.covariance = _F @ self.covariance @ _F.T + q
        return self.bbox

    def update(self, bbox: np.ndarray, is_fast: bool) -> None:
        z = _bbox_to_state(bbox)[:4]
        innovation = z - (_H @ self.state)
        r = _R_FAST if is_fast else _R
        s = _H @ self.covariance @ _H.T + r
        k = self.covariance @ _H.T @ np.linalg.inv(s)
        self.state = self.state + (k @ innovation)
        self.covariance = (_I - k @ _H) @ self.covariance
        self.missed = 0
        self.hits += 1

    @property
    def bbox(self) -> np.ndarray:
        return _state_to_bbox(self.state)

    @property
    def velocity_magnitude(self) -> float:
        """Kalman velocity magnitude in normalized units/frame — the signal
        TEMPORAL_ANOMALY checks: a "cone" track that's still moving after
        confirmation is anatomically attached to something that walks."""
        return float(np.hypot(self.state[4], self.state[5]))


class SortTracker:
    """Call `update()` once per frame (in frame order) with that frame's raw
    detections; returns the same detections annotated with `track_id` (and
    Kalman-smoothed coordinates), plus synthesized "predicted" detections
    for briefly-occluded tracks. `hits`/`missed`/`velocity_magnitude` are
    read back via `track_stats()` for the quality rules that need them —
    the tracker itself has no notion of "suspicious", it just tracks.
    """

    def __init__(
        self,
        static_class_ids: frozenset[int] = frozenset(),
        fast_class_ids: frozenset[int] = frozenset(),
        iou_threshold: float = 0.25,
        max_missed: int = 16,
        high_conf: float = 0.45,
        low_conf: float = 0.10,
    ):
        self._static_ids = static_class_ids
        self._fast_ids = fast_class_ids
        self._iou_threshold = iou_threshold
        self._max_missed = max_missed
        self._high_conf = high_conf
        self._low_conf = low_conf
        self._next_id = 1
        self._tracks: List[_Track] = []

    def _is_static(self, class_id: int) -> bool:
        return class_id in self._static_ids

    def _is_fast(self, class_id: int) -> bool:
        return class_id in self._fast_ids

    def _max_predict(self, class_id: int) -> int:
        if self._is_static(class_id):
            return _STATIC_PREDICT_MAX_MISSED
        if self._is_fast(class_id):
            return _FAST_PREDICT_MAX_MISSED
        return _DEFAULT_PREDICT_MAX_MISSED

    def _min_hits(self, class_id: int) -> int:
        if self._is_static(class_id):
            return _STATIC_PREDICT_MIN_HITS
        if self._is_fast(class_id):
            return _FAST_PREDICT_MIN_HITS
        return _DEFAULT_PREDICT_MIN_HITS

    def _class_high_conf(self, det: Dict) -> float:
        class_id = int(det.get("class_id", -1))
        if self._is_fast(class_id):
            return min(self._high_conf, _FAST_HIGH_CONF_CAP)
        return self._high_conf

    def update(self, detections: List[Dict]) -> List[Dict]:
        for track in self._tracks:
            track.predict(self._is_static(track.class_id), self._is_fast(track.class_id))
            track.missed += 1

        high = [d for d in detections if float(d.get("confidence", 0.0)) >= self._class_high_conf(d)]
        low = [
            d for d in detections if self._low_conf <= float(d.get("confidence", 0.0)) < self._class_high_conf(d)
        ]

        unmatched_tracks = set(range(len(self._tracks)))
        matched_tracks, _ = self._match(high, unmatched_tracks)
        unmatched_tracks -= matched_tracks
        matched_tracks_low, _ = self._match(low, unmatched_tracks)
        unmatched_tracks -= matched_tracks_low

        for det in self._new_track_candidates(high):
            if "track_id" in det:
                continue
            track = _Track(
                track_id=self._next_id,
                class_id=int(det["class_id"]),
                state=_bbox_to_state(_bbox(det)),
                covariance=np.diag([1.0, 1.0, 1.0, 1.0, 1e4, 1e4, 1e4, 1e4]).astype(np.float32),
                missed=0,
                hits=1,
            )
            self._next_id += 1
            self._tracks.append(track)
            det["track_id"] = track.track_id

        for det in detections:
            tid = det.get("track_id")
            if tid is None:
                continue
            track = next((t for t in self._tracks if t.track_id == tid), None)
            if track is None or track.hits < 2:
                continue
            sb = track.bbox
            det["x1"], det["y1"], det["x2"], det["y2"] = (float(sb[0]), float(sb[1]), float(sb[2]), float(sb[3]))
            det["cx"] = float((sb[0] + sb[2]) * 0.5)
            det["cy"] = float((sb[1] + sb[3]) * 0.5)

        matched_ids = {d.get("track_id") for d in detections if "track_id" in d}
        for track in self._tracks:
            if track.track_id in matched_ids:
                continue
            if track.missed > self._max_predict(track.class_id) or track.hits < self._min_hits(track.class_id):
                continue
            sb = track.bbox
            detections.append(
                {
                    "class_id": track.class_id,
                    "confidence": 0.2,
                    "raw_detected": False,
                    "x1": float(sb[0]),
                    "y1": float(sb[1]),
                    "x2": float(sb[2]),
                    "y2": float(sb[3]),
                    "cx": float((sb[0] + sb[2]) * 0.5),
                    "cy": float((sb[1] + sb[3]) * 0.5),
                    "track_id": track.track_id,
                    "predicted": True,
                }
            )

        self._tracks = [t for t in self._tracks if t.missed <= max(self._max_missed, self._max_predict(t.class_id))]
        return detections

    def track_stats(self) -> dict[int, dict]:
        """Current state of every live track — `{track_id: {hits, missed,
        velocity_magnitude, class_id}}`. Read by the quality rules after a
        whole clip has been fed through `update()` frame by frame."""
        return {
            t.track_id: {"hits": t.hits, "missed": t.missed, "velocity_magnitude": t.velocity_magnitude, "class_id": t.class_id}
            for t in self._tracks
        }

    def _new_track_candidates(self, detections: List[Dict]) -> List[Dict]:
        candidates = [d for d in detections if "track_id" not in d]
        non_fast = [d for d in candidates if int(d.get("class_id", -1)) not in self._fast_ids]
        fast = [d for d in candidates if int(d.get("class_id", -1)) in self._fast_ids]
        if not fast:
            return non_fast
        # at most one new track per fast-class per frame (mirrors the source's ball-specific cap)
        by_class: dict[int, Dict] = {}
        for d in fast:
            cid = int(d["class_id"])
            if cid not in by_class or float(d.get("confidence", 0.0)) > float(by_class[cid].get("confidence", 0.0)):
                by_class[cid] = d
        return non_fast + list(by_class.values())

    def _match(self, detections: List[Dict], track_indices: set[int]) -> tuple[set[int], set[int]]:
        assigned_tracks: set[int] = set()
        assigned_dets: set[int] = set()
        if not detections or not track_indices:
            return assigned_tracks, assigned_dets

        pairs: list[tuple[float, int, int]] = []
        for ti in track_indices:
            track = self._tracks[ti]
            pred_bbox = track.bbox
            pred_center = np.array([(pred_bbox[0] + pred_bbox[2]) * 0.5, (pred_bbox[1] + pred_bbox[3]) * 0.5], dtype=np.float32)
            for di, det in enumerate(detections):
                if int(track.class_id) != int(det["class_id"]):
                    continue
                det_bbox = _bbox(det)
                det_center = np.array([(det_bbox[0] + det_bbox[2]) * 0.5, (det_bbox[1] + det_bbox[3]) * 0.5], dtype=np.float32)
                center_dist = float(np.linalg.norm(pred_center - det_center))
                iou = _iou(pred_bbox, det_bbox)

                if self._is_fast(track.class_id):
                    size_gate = 4.0 * max(float(pred_bbox[2] - pred_bbox[0]), float(pred_bbox[3] - pred_bbox[1]))
                    dist_gate = max(_CENTER_DIST_GATE_FAST, min(0.24, size_gate))
                else:
                    dist_gate = _CENTER_DIST_GATE

                if iou >= self._iou_threshold:
                    score = iou - 0.02 * center_dist
                elif center_dist < dist_gate:
                    score = 0.1 - center_dist
                else:
                    continue
                pairs.append((score, ti, di))

        for _score, ti, di in sorted(pairs, reverse=True):
            if ti in assigned_tracks or di in assigned_dets:
                continue
            self._tracks[ti].update(_bbox(detections[di]), self._is_fast(self._tracks[ti].class_id))
            detections[di]["track_id"] = self._tracks[ti].track_id
            assigned_tracks.add(ti)
            assigned_dets.add(di)

        return assigned_tracks, assigned_dets
