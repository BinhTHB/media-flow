"""
Subtitle Event Segmenter

Splits spatial subtitle tracks into distinct semantic temporal SubtitleEvents.
Prevents two consecutive captions appearing at the same position from being merged into one event.

Pure Python logic + frame/evidence discontinuity splitting.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from src.core.models.bbox import BBox
from src.core.models.detection import Detection
from src.core.models.subtitle_event import SubtitleEvent
from src.core.models.track import Track

logger = logging.getLogger(__name__)


class SubtitleEventSegmenter:
    def __init__(self, max_gap_seconds: float = 0.5, bbox_discontinuity_threshold: float = 0.40):
        self.max_gap_seconds = max_gap_seconds
        self.bbox_discontinuity_threshold = bbox_discontinuity_threshold

    def segment_track(self, track: Track) -> List[SubtitleEvent]:
        """Split a spatial Track into one or more SubtitleEvents."""
        if not track.detections:
            return []

        events: List[SubtitleEvent] = []
        event_counter = 1

        # Group detections by gap or visual discontinuity
        current_group: List[Detection] = [track.detections[0]]

        for i in range(1, len(track.detections)):
            prev = track.detections[i - 1]
            curr = track.detections[i]

            gap = curr.timestamp_seconds - prev.timestamp_seconds
            iou = prev.bbox.iou(curr.bbox)

            # Discontinuity check: gap > max_gap or bbox IoU drops significantly
            if gap > self.max_gap_seconds or iou < self.bbox_discontinuity_threshold:
                # Close current event
                events.append(self._build_event(track.track_id, event_counter, current_group))
                event_counter += 1
                current_group = [curr]
            else:
                current_group.append(curr)

        if current_group:
            events.append(self._build_event(track.track_id, event_counter, current_group))

        return events

    def _build_event(self, track_id: str, index: int, detections: List[Detection]) -> SubtitleEvent:
        start_time = detections[0].timestamp_seconds
        end_time = detections[-1].timestamp_seconds
        if end_time <= start_time:
            end_time = start_time + 0.1

        mid_idx = len(detections) // 2
        anchor_bbox = detections[mid_idx].bbox
        trajectory = [d.bbox for d in detections]

        event_id = f"{track_id}_E{index:02d}"
        return SubtitleEvent(
            event_id=event_id,
            track_id=track_id,
            start_time=start_time,
            end_time=end_time,
            anchor_bbox=anchor_bbox,
            bbox_trajectory=trajectory,
        )
