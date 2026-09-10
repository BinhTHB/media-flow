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
        content_change_threshold: float = 0.18,
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

        # A tracker frame can contain several horizontally-adjacent YOLO fragments
        # belonging to the same subtitle row. Temporal segmentation must operate on
        # one observation per video frame; otherwise sampling every N detections
        # advances by an unpredictable fraction of a frame and misses real caption
        # changes while manufacturing micro-events.
        detections = self._collapse_same_frame_detections(track.detections)

        # Step 1: Primary temporal gap & bbox discontinuity splitting
        initial_groups: List[Tuple[str, List[Detection]]] = []
        current_group: List[Detection] = [detections[0]]
        current_split_reason = "initial"

        n = len(detections)
        i = 1

        while i < n:
            prev = detections[i - 1]
            curr = detections[i]

            gap = curr.timestamp_seconds - prev.timestamp_seconds
            iou = prev.bbox.iou(curr.bbox)

            split_reason = None
            if gap > self.max_gap_seconds:
                split_reason = "temporal_gap"
            elif iou < self.bbox_discontinuity_threshold:
                persists = True
                for lookahead in range(1, min(self.min_discontinuity_frames, n - i)):
                    ahead_det = detections[i + lookahead]
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

        # Step 3: Keep content boundaries intact. A previous generic micro-fragment
        # merger used an OR-condition (current OR previous is short), which could
        # chain-merge a whole sequence of genuine caption changes into one multi-
        # second event. Duplicate fragments are now merged safely *after OCR*, when
        # semantic equality is known.
        merged_groups = final_groups

        # Step 4: Build SubtitleEvent objects
        events: List[SubtitleEvent] = []
        for idx, (reason, group_dets) in enumerate(merged_groups, start=1):
            events.append(self._build_event(track.track_id, idx, group_dets, reason))

        return events

    @staticmethod
    def _collapse_same_frame_detections(detections: List[Detection]) -> List[Detection]:
        if not detections:
            return []
        by_frame: dict[int, List[Detection]] = {}
        for det in detections:
            by_frame.setdefault(int(det.frame_index), []).append(det)

        collapsed: List[Detection] = []
        for frame_index in sorted(by_frame):
            group = by_frame[frame_index]
            boxes = [d.bbox for d in group]
            box = BBox(
                x1=min(b.x1 for b in boxes),
                y1=min(b.y1 for b in boxes),
                x2=max(b.x2 for b in boxes),
                y2=max(b.y2 for b in boxes),
                frame_width=boxes[0].frame_width,
                frame_height=boxes[0].frame_height,
            )
            first = group[0]
            collapsed.append(
                Detection(
                    class_id=first.class_id,
                    class_name=first.class_name,
                    confidence=max(float(d.confidence) for d in group),
                    bbox=box,
                    frame_index=frame_index,
                    timestamp_seconds=min(float(d.timestamp_seconds) for d in group),
                    candidate_id="+".join(d.candidate_id for d in group if d.candidate_id),
                )
            )
        return collapsed

    def _split_by_content_change(
        self, detections: List[Detection], video_path: str
    ) -> List[Tuple[str, List[Detection]]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [("initial", detections)]

        n = len(detections)
        split_indices = [0]

        # Compare all samples through one stable crop window. Resizing each
        # detection's own bbox independently normalizes away exactly the spatial
        # changes that distinguish one subtitle sentence from the next.
        boxes = [d.bbox for d in detections]
        fw = boxes[0].frame_width
        fh = boxes[0].frame_height
        centers_x = sorted((b.x1 + b.x2) / 2.0 for b in boxes)
        center_x = centers_x[len(centers_x) // 2]
        widths = sorted(float(b.width) for b in boxes)
        median_w = widths[len(widths) // 2]
        # A long sentence elsewhere in the same spatial track must not make the
        # change-detection crop enormous. We only need a stable central glyph
        # window, not the full caption width.
        stable_w = min(float(fw) * 0.52, max(median_w * 1.20, float(fw) * 0.28))

        y1_values = sorted(float(b.y1) for b in boxes)
        y2_values = sorted(float(b.y2) for b in boxes)
        p10_index = int(round((len(boxes) - 1) * 0.10))
        p90_y_index = int(round((len(boxes) - 1) * 0.90))
        y1 = max(0.0, y1_values[p10_index] - 8.0)
        y2 = min(float(fh), y2_values[p90_y_index] + 8.0)
        stable_bbox = BBox(
            x1=max(0.0, center_x - stable_w / 2.0),
            y1=y1,
            x2=min(float(fw), center_x + stable_w / 2.0),
            y2=y2,
            frame_width=fw,
            frame_height=fh,
        )

        prev_norm = None
        prev_local_norm = None
        prev_sample_det = None
        i = 0

        while i < n:
            det = detections[i]
            cap.set(cv2.CAP_PROP_POS_FRAMES, det.frame_index)
            ret, frame = cap.read()
            if ret and frame is not None:
                norm = _extract_norm_crop(frame, stable_bbox)
                local_norm = _extract_norm_crop(frame, det.bbox)
                if norm is not None and prev_norm is not None:
                    stable_diff = float(np.mean(np.abs(norm.astype(float) - prev_norm.astype(float))) / 255.0)
                    local_diff = 0.0
                    if local_norm is not None and prev_local_norm is not None:
                        local_diff = float(np.mean(np.abs(local_norm.astype(float) - prev_local_norm.astype(float))) / 255.0)
                    width_ratio = 0.0
                    if prev_sample_det is not None:
                        prev_w = max(1.0, float(prev_sample_det.bbox.width))
                        width_ratio = abs(float(det.bbox.width) - prev_w) / prev_w

                    # Strong changes in a stable screen-space crop are reliable
                    # caption boundaries. A weaker stable change is accepted only
                    # when the local glyph crop changes strongly AND the detected
                    # row width changes materially. This catches short→long caption
                    # switches without reacting to compression/animation noise.
                    strong_change = stable_diff >= self.content_change_threshold
                    supported_change = (
                        stable_diff >= 0.12
                        and local_diff >= 0.23
                        and width_ratio >= 0.15
                    )
                    if strong_change or supported_change:
                        # Sampling is normally every 3 frames (~100 ms at 30 fps),
                        # already finer than viewer-perceptible subtitle timing. Do
                        # not perform noisy frame-by-frame refinement here; the
                        # boundary refiner later tightens first/last matching frames.
                        if i > split_indices[-1]:
                            split_indices.append(i)
                if norm is not None:
                    prev_norm = norm
                if local_norm is not None:
                    prev_local_norm = local_norm
                prev_sample_det = det
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
        trajectory_frames = [int(d.frame_index) for d in detections]

        event_id = f"{track_id}_E{index:02d}"
        return SubtitleEvent(
            event_id=event_id,
            track_id=track_id,
            start_time=start_time,
            end_time=end_time,
            anchor_bbox=anchor_bbox,
            bbox_trajectory=trajectory,
            bbox_frame_indices=trajectory_frames,
            split_reason=split_reason,
        )
