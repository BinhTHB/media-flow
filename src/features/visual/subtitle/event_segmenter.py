"""
Subtitle Event Segmenter

Splits spatial subtitle tracks into distinct semantic temporal SubtitleEvents.
Uses debounced temporal and visual evidence to prevent detector jitter from creating micro-events,
while ensuring genuine consecutive captions at the same location are split.

Pure Python logic, zero third-party dependencies.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.core.models.bbox import BBox
from src.core.models.detection import Detection
from src.core.models.subtitle_event import SubtitleEvent
from src.core.models.track import Track

logger = logging.getLogger(__name__)


def _extract_norm_crop(frame: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox.xyxy_int()
    pad_x, pad_y = 10, 6
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(frame.shape[1], x2 + pad_x)
    y2 = min(frame.shape[0], y2 + pad_y)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    res = cv2.resize(gray, (192, 64))
    return cv2.equalizeHist(res)


class SubtitleEventSegmenter:
    def __init__(
        self,
        max_gap_seconds: float = 0.5,
        bbox_discontinuity_threshold: float = 0.15,
        min_discontinuity_frames: int = 3,
        min_event_duration: float = 0.25,
        content_sample_stride_frames: int = 3,
        content_change_threshold: float = 0.22,
    ):
        self.max_gap_seconds = max_gap_seconds
        self.bbox_discontinuity_threshold = bbox_discontinuity_threshold
        self.min_discontinuity_frames = min_discontinuity_frames
        self.min_event_duration = min_event_duration
        self.content_sample_stride_frames = content_sample_stride_frames
        self.content_change_threshold = content_change_threshold

    def segment_tracks(
        self, tracks: List[Track], video_path: Optional[str] = None
    ) -> List[SubtitleEvent]:
        events: List[SubtitleEvent] = []
        for track in tracks:
            events.extend(self.segment_track(track, video_path=video_path))
        return events

    def segment_track(
        self, track: Track, video_path: Optional[str] = None
    ) -> List[SubtitleEvent]:
        """Split a spatial Track into one or more SubtitleEvents."""
        if not track.detections:
            return []

        # Step 1: Primary temporal gap & bbox discontinuity splitting
        initial_groups: List[Tuple[str, List[Detection]]] = []
        current_group: List[Detection] = [track.detections[0]]
        current_split_reason = "initial"

        n = len(track.detections)
        i = 1

        while i < n:
            prev = track.detections[i - 1]
            curr = track.detections[i]

            gap = curr.timestamp_seconds - prev.timestamp_seconds
            iou = prev.bbox.iou(curr.bbox)

            split_reason = None
            if gap > self.max_gap_seconds:
                split_reason = "temporal_gap"
            elif iou < self.bbox_discontinuity_threshold:
                persists = True
                for lookahead in range(1, min(self.min_discontinuity_frames, n - i)):
                    ahead_det = track.detections[i + lookahead]
                    if prev.bbox.iou(ahead_det.bbox) >= self.bbox_discontinuity_threshold:
                        persists = False
                        break
                if persists:
                    split_reason = "bbox_discontinuity"

            if split_reason is not None:
                initial_groups.append((current_split_reason, current_group))
                current_group = [curr]
                current_split_reason = split_reason
            else:
                current_group.append(curr)

            i += 1

        if current_group:
            initial_groups.append((current_split_reason, current_group))

        # Step 2: Secondary Visual Content-Change detection inside each group
        final_groups: List[Tuple[str, List[Detection]]] = []
        for reason, group_dets in initial_groups:
            if video_path and len(group_dets) >= 6:
                content_splits = self._split_by_content_change(group_dets, video_path)
                for c_reason, c_dets in content_splits:
                    final_groups.append((c_reason if c_reason != "initial" else reason, c_dets))
            else:
                final_groups.append((reason, group_dets))

        # Step 3: Merge micro-fragments that were split due to transient noise
        merged_groups = self._merge_micro_fragments(final_groups)

        # Step 4: Build SubtitleEvent objects
        events: List[SubtitleEvent] = []
        for idx, (reason, group_dets) in enumerate(merged_groups, start=1):
            events.append(self._build_event(track.track_id, idx, group_dets, reason))

        return events

    def _split_by_content_change(
        self, detections: List[Detection], video_path: str
    ) -> List[Tuple[str, List[Detection]]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [("initial", detections)]

        n = len(detections)
        split_indices = [0]

        prev_norm = None
        i = 0

        while i < n:
            det = detections[i]
            cap.set(cv2.CAP_PROP_POS_FRAMES, det.frame_index)
            ret, frame = cap.read()
            if ret and frame is not None:
                norm = _extract_norm_crop(frame, det.bbox)
                if norm is not None and prev_norm is not None:
                    diff = float(np.mean(np.abs(norm.astype(float) - prev_norm.astype(float))) / 255.0)
                    if diff >= self.content_change_threshold:
                        # Refine boundary in range [i - stride, i]
                        refined_idx = i
                        for r_idx in range(max(1, i - self.content_sample_stride_frames), i + 1):
                            r_det = detections[r_idx]
                            cap.set(cv2.CAP_PROP_POS_FRAMES, r_det.frame_index)
                            r_ret, r_frame = cap.read()
                            if r_ret and r_frame is not None:
                                r_norm = _extract_norm_crop(r_frame, r_det.bbox)
                                if r_norm is not None and prev_norm is not None:
                                    r_diff = float(np.mean(np.abs(r_norm.astype(float) - prev_norm.astype(float))) / 255.0)
                                    if r_diff >= 0.18:
                                        refined_idx = r_idx
                                        break
                        if refined_idx > split_indices[-1]:
                            split_indices.append(refined_idx)
                if norm is not None:
                    prev_norm = norm
            i += self.content_sample_stride_frames

        cap.release()

        split_indices.append(n)
        split_indices = sorted(list(set(split_indices)))

        results: List[Tuple[str, List[Detection]]] = []
        for idx in range(len(split_indices) - 1):
            s_i = split_indices[idx]
            e_i = split_indices[idx + 1]
            sub_dets = detections[s_i:e_i]
            if sub_dets:
                reason = "initial" if idx == 0 else "content_change"
                results.append((reason, sub_dets))

        return results

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
