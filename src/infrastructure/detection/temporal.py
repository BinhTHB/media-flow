"""Lightweight IoU-based temporal stabilization for video detections.

The fine-tuned model is good at detecting dynamic subtitles, but a video can
miss a few frames here and there. This module:

* matches boxes across consecutive frames using IoU + class;
* carries a track through short gaps (1-2 frames);
* smooths box coordinates (exponential moving average);
* drops a track promptly when the object disappears;
* never mixes tracks of different classes;
* adds negligible latency and can be disabled entirely.

The implementation is deliberately simple (no heavyweight multi-object
tracker) because IoU-based matching is sufficient for stable screen text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from src.infrastructure.detection.postprocess import box_iou
from src.infrastructure.detection.schemas import Detection


@dataclass
class Track:
    """A stabilized track of detections of a single class."""

    class_id: int
    class_name: str
    last_frame: int
    frames_since_update: int = 0
    boxes: List[List[float]] = field(default_factory=list)  # smoothed boxes
    frame_indices: List[int] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    track_id: str = ""


def _average(a: List[float], b: List[float], alpha: float) -> List[float]:
    return [alpha * av + (1.0 - alpha) * bv for av, bv in zip(a, b)]


class TemporalStabilizer:
    """Stabilize per-frame detections across time using IoU matching."""

    def __init__(
        self,
        enabled: bool = True,
        max_gap_frames: int = 2,
        min_track_frames: int = 2,
        iou_match_threshold: float = 0.40,
        box_smoothing_alpha: float = 0.65,
    ) -> None:
        self.enabled = enabled
        self.max_gap_frames = int(max_gap_frames)
        self.min_track_frames = int(min_track_frames)
        self.iou_match_threshold = float(iou_match_threshold)
        self.box_smoothing_alpha = float(box_smoothing_alpha)
        self._tracks: List[Track] = []
        self._track_counter = 0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._tracks = []
        self._track_counter = 0

    # ------------------------------------------------------------------
    def _best_match(
        self, det: Detection, tracks: Sequence[Track]
    ) -> Optional[Track]:
        best: Optional[Track] = None
        best_iou = self.iou_match_threshold
        for track in tracks:
            if track.class_id != det.class_id:
                continue
            # cannot match a track that has been waiting past the max gap
            if track.frames_since_update > self.max_gap_frames:
                continue
            last_box = track.boxes[-1]
            iou = box_iou(last_box, det.bbox_xyxy)
            if iou >= best_iou:
                best = track
                best_iou = iou
        return best

    # ------------------------------------------------------------------
    def stabilize(
        self, frame_index: int, detections: Sequence[Detection]
    ) -> List[Detection]:
        """Process detections for one frame, returning stabilized detections."""
        if not self.enabled:
            return list(detections)

        for track in self._tracks:
            track.frames_since_update += 1

        matched_tracks: List[Track] = []
        kept: List[Detection] = []
        for det in detections:
            track = self._best_match(det, self._tracks)
            if track is not None:
                track.frames_since_update = 0
                track.last_frame = frame_index
                prev = track.boxes[-1]
                new_box = _average(
                    det.bbox_xyxy, prev, self.box_smoothing_alpha
                )
                track.boxes.append(new_box)
                track.frame_indices.append(frame_index)
                track.confidences.append(det.confidence)
                if track not in matched_tracks:
                    matched_tracks.append(track)
                kept.append(
                    det._with_bbox(*new_box)
                )
            else:
                # start a new track for this detection
                self._track_counter += 1
                new_track = Track(
                    class_id=det.class_id,
                    class_name=det.class_name,
                    last_frame=frame_index,
                    boxes=[list(det.bbox_xyxy)],
                    frame_indices=[frame_index],
                    confidences=[det.confidence],
                    track_id=f"T{self._track_counter:06d}_{det.class_name}",
                )
                self._tracks.append(new_track)
                if new_track not in matched_tracks:
                    matched_tracks.append(new_track)
                kept.append(det)

        # Drop stale tracks and short-lived tracks.
        alive: List[Track] = []
        for track in self._tracks:
            if track in matched_tracks:
                alive.append(track)
            elif track.frames_since_update <= self.max_gap_frames:
                # keep injecting the last known box through the gap so subtitles
                # that briefly disappear remain stable (no new Detection here,
                # but track stays alive for future frames).
                alive.append(track)
        self._tracks = alive

        # Apply min_track_frames: only emit detections that belong to a track
        # that has existed for at least min_track_frames observations.
        if self.min_track_frames > 1:
            stable: List[Detection] = []
            for det in kept:
                track = self._find_track(det)
                if track is not None and len(track.frame_indices) >= self.min_track_frames:
                    stable.append(det)
            kept = stable
        return kept

    # ------------------------------------------------------------------
    def _find_track(self, det: Detection) -> Optional[Track]:
        # Match by class and by the last emitted box (closest in time).
        best: Optional[Track] = None
        best_iou = self.iou_match_threshold
        for track in self._tracks:
            if track.class_id != det.class_id:
                continue
            if not track.boxes:
                continue
            iou = box_iou(track.boxes[-1], det.bbox_xyxy)
            if iou >= best_iou:
                best = track
                best_iou = iou
        return best

    def to_dict(self) -> Dict[str, object]:
        """Snapshot of active tracks (for debugging / serialization)."""
        return {
            "active_tracks": len(self._tracks),
            "tracks": [
                {
                    "track_id": t.track_id,
                    "class_name": t.class_name,
                    "last_frame": t.last_frame,
                    "frames_since_update": t.frames_since_update,
                    "n_frames": len(t.frame_indices),
                }
                for t in self._tracks
            ],
        }
