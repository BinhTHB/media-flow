"""
Subtitle Event Segmenter

Splits spatial subtitle tracks into distinct semantic temporal SubtitleEvents.
Uses debounced temporal and visual evidence to prevent detector jitter from creating micro-events,
while ensuring genuine consecutive captions at the same location are split.

Pure Python logic, zero third-party dependencies.
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
    def __init__(
        self,
        max_gap_seconds: float = 0.5,
        bbox_discontinuity_threshold: float = 0.15,
        min_discontinuity_frames: int = 3,
        min_event_duration: float = 0.25,
    ):
        self.max_gap_seconds = max_gap_seconds
        self.bbox_discontinuity_threshold = bbox_discontinuity_threshold
        self.min_discontinuity_frames = min_discontinuity_frames
        self.min_event_duration = min_event_duration

    def segment_track(self, track: Track) -> List[SubtitleEvent]:
        """Split a spatial Track into one or more SubtitleEvents."""
        if not track.detections:
            return []

        events_groups: List[Tuple[str, List[Detection]]] = []
        current_group: List[Detection] = [track.detections[0]]
        current_split_reason = "initial"

        i = 1
        n = len(track.detections)

        while i < n:
            prev = track.detections[i - 1]
            curr = track.detections[i]

            gap = curr.timestamp_seconds - prev.timestamp_seconds
            iou = prev.bbox.iou(curr.bbox)

            split_reason = None
            if gap > self.max_gap_seconds:
                split_reason = "temporal_gap"
            elif iou < self.bbox_discontinuity_threshold:
                # Check if the low IoU persists across min_discontinuity_frames
                persists = True
                for lookahead in range(1, min(self.min_discontinuity_frames, n - i)):
                    ahead_det = track.detections[i + lookahead]
                    if prev.bbox.iou(ahead_det.bbox) >= self.bbox_discontinuity_threshold:
                        persists = False
                        break
                if persists:
                    split_reason = "bbox_discontinuity"

            if split_reason is not None:
                events_groups.append((current_split_reason, current_group))
                current_group = [curr]
                current_split_reason = split_reason
            else:
                current_group.append(curr)

            i += 1

        if current_group:
            events_groups.append((current_split_reason, current_group))

        # Merge micro-fragments that were split due to transient noise
        merged_groups = self._merge_micro_fragments(events_groups)

        # Build SubtitleEvent objects
        events: List[SubtitleEvent] = []
        for idx, (reason, group_dets) in enumerate(merged_groups, start=1):
            events.append(self._build_event(track.track_id, idx, group_dets, reason))

        return events

    def _merge_micro_fragments(
        self, groups: List[Tuple[str, List[Detection]]]
    ) -> List[Tuple[str, List[Detection]]]:
        if len(groups) <= 1:
            return groups

        merged: List[Tuple[str, List[Detection]]] = [groups[0]]

        for i in range(1, len(groups)):
            reason, curr_dets = groups[i]
            prev_reason, prev_dets = merged[-1]

            prev_duration = prev_dets[-1].timestamp_seconds - prev_dets[0].timestamp_seconds
            curr_duration = curr_dets[-1].timestamp_seconds - curr_dets[0].timestamp_seconds
            gap = curr_dets[0].timestamp_seconds - prev_dets[-1].timestamp_seconds

            # Merge if current group is a micro-fragment and gap is small
            if (curr_duration < self.min_event_duration or prev_duration < self.min_event_duration) and gap <= 0.35:
                merged[-1] = (prev_reason, prev_dets + curr_dets)
            else:
                merged.append((reason, curr_dets))

        return merged

    def _build_event(
        self, track_id: str, index: int, detections: List[Detection], split_reason: str
    ) -> SubtitleEvent:
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
            split_reason=split_reason,
        )
