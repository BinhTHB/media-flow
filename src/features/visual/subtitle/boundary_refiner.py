"""Frame-level subtitle boundary refinement.

YOLO gives a strong coarse interval but can miss a few leading/trailing frames.
This module searches only a small neighborhood around each event and extends the
interval when adjacent frames still visually match the detected subtitle. It does
not blindly add fixed milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.core.models.bbox import BBox
from src.core.models.subtitle_event import SubtitleEvent


@dataclass(frozen=True)
class BoundaryRefinementConfig:
    scan_back_seconds: float = 0.40
    scan_forward_seconds: float = 0.40
    present_threshold: float = 0.45
    min_glyph_agreement: float = 0.84
    absence_confirm_frames: int = 2
    bbox_pad_ratio_x: float = 0.20
    bbox_pad_ratio_y: float = 0.25


class SubtitleBoundaryRefiner:
    def __init__(self, config: Optional[BoundaryRefinementConfig] = None) -> None:
        self.config = config or BoundaryRefinementConfig()

    def refine(self, events: List[SubtitleEvent], video_path: str) -> List[SubtitleEvent]:
        if not events or not video_path:
            return events

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return events
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            cap.release()
            return events

        coarse: dict[str, Tuple[int, int]] = {}
        for event in events:
            start_frame, end_frame = self._event_frame_bounds(event, fps)
            coarse[event.event_id] = (start_frame, end_frame)
            start_frame = self._refine_start(cap, event, start_frame, fps)
            end_frame = self._refine_end(cap, event, end_frame, fps, total_frames)
            if end_frame < start_frame:
                start_frame, end_frame = coarse[event.event_id]
            event.start_time = max(0.0, start_frame / fps)
            event.end_time = max(event.start_time + (1.0 / fps), end_frame / fps)

        cap.release()
        self._clamp_same_track_collisions(events, coarse, fps)
        return events

    @staticmethod
    def _event_frame_bounds(event: SubtitleEvent, fps: float) -> Tuple[int, int]:
        frame_indices = getattr(event, "bbox_frame_indices", None) or []
        if frame_indices:
            return int(frame_indices[0]), int(frame_indices[-1])
        return int(round(event.start_time * fps)), int(round(event.end_time * fps))

    def _refine_start(self, cap: cv2.VideoCapture, event: SubtitleEvent, start_frame: int, fps: float) -> int:
        ref_bbox = self._boundary_bbox(event, first=True)
        ref = self._normalized_crop(cap, start_frame, ref_bbox)
        if ref is None:
            return start_frame
        max_scan = max(1, int(round(self.config.scan_back_seconds * fps)))
        earliest = start_frame
        absent_run = 0
        for frame_index in range(start_frame - 1, max(-1, start_frame - max_scan - 1), -1):
            if frame_index < 0:
                break
            candidate = self._normalized_crop(cap, frame_index, ref_bbox)
            score = self._similarity(ref, candidate)
            if score >= self.config.present_threshold:
                earliest = frame_index
                absent_run = 0
            else:
                absent_run += 1
                if absent_run >= self.config.absence_confirm_frames:
                    break
        return earliest

    def _refine_end(
        self,
        cap: cv2.VideoCapture,
        event: SubtitleEvent,
        end_frame: int,
        fps: float,
        total_frames: int,
    ) -> int:
        ref_bbox = self._boundary_bbox(event, first=False)
        ref = self._normalized_crop(cap, end_frame, ref_bbox)
        if ref is None:
            return end_frame
        max_scan = max(1, int(round(self.config.scan_forward_seconds * fps)))
        latest = end_frame
        absent_run = 0
        upper = min(total_frames - 1, end_frame + max_scan)
        for frame_index in range(end_frame + 1, upper + 1):
            candidate = self._normalized_crop(cap, frame_index, ref_bbox)
            score = self._similarity(ref, candidate)
            if score >= self.config.present_threshold:
                latest = frame_index
                absent_run = 0
            else:
                absent_run += 1
                if absent_run >= self.config.absence_confirm_frames:
                    break
        return latest

    @staticmethod
    def _boundary_bbox(event: SubtitleEvent, *, first: bool) -> BBox:
        if event.bbox_trajectory:
            return event.bbox_trajectory[0 if first else -1]
        return event.anchor_bbox

    def _normalized_crop(
        self,
        cap: cv2.VideoCapture,
        frame_index: int,
        bbox: BBox,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            return None

        pad_x = max(8, int(round(bbox.width * self.config.bbox_pad_ratio_x)))
        pad_y = max(6, int(round(bbox.height * self.config.bbox_pad_ratio_y)))
        x1, y1, x2, y2 = bbox.xyxy_int()
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(frame.shape[1], x2 + pad_x)
        y2 = min(frame.shape[0], y2 + pad_y)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (192, 64), interpolation=cv2.INTER_AREA)
        gray = cv2.equalizeHist(gray)
        edges = cv2.Canny(gray, 60, 160)
        return gray, edges

    def _similarity(
        self,
        reference: Optional[Tuple[np.ndarray, np.ndarray]],
        candidate: Optional[Tuple[np.ndarray, np.ndarray]],
    ) -> float:
        if reference is None or candidate is None:
            return -1.0
        ref_gray, ref_edges = reference
        cand_gray, cand_edges = candidate
        gray_corr = float(cv2.matchTemplate(cand_gray, ref_gray, cv2.TM_CCOEFF_NORMED)[0, 0])
        if not np.isfinite(gray_corr):
            gray_corr = 0.0

        # Fuzzy edge overlap focuses on glyph shape instead of static background.
        # A one-pixel dilation tolerates tiny compression/jitter shifts, while a
        # different caption in the same box still has much lower symmetric overlap.
        kernel = np.ones((3, 3), dtype=np.uint8)
        ref_mask = ref_edges > 0
        cand_mask = cand_edges > 0
        ref_count = int(ref_mask.sum())
        cand_count = int(cand_mask.sum())
        if ref_count == 0 or cand_count == 0:
            edge_overlap = 0.0
            glyph_agreement = 0.0
        else:
            ref_dilated = cv2.dilate(ref_edges, kernel, iterations=1) > 0
            cand_dilated = cv2.dilate(cand_edges, kernel, iterations=1) > 0
            ref_match = float(np.logical_and(ref_mask, cand_dilated).sum()) / ref_count
            cand_match = float(np.logical_and(cand_mask, ref_dilated).sum()) / cand_count
            edge_overlap = 0.5 * (ref_match + cand_match)

            glyph_union = np.logical_or(ref_dilated, cand_dilated)
            pixel_diff = np.abs(ref_gray.astype(np.float32) - cand_gray.astype(np.float32)) / 255.0
            glyph_agreement = (
                float(1.0 - pixel_diff[glyph_union].mean())
                if glyph_union.any()
                else 0.0
            )

        if glyph_agreement < self.config.min_glyph_agreement:
            return 0.0
        return (0.20 * max(0.0, gray_corr)) + (0.80 * edge_overlap)

    @staticmethod
    def _clamp_same_track_collisions(
        events: List[SubtitleEvent],
        coarse: dict[str, Tuple[int, int]],
        fps: float,
    ) -> None:
        tracks: dict[str, List[SubtitleEvent]] = {}
        for event in events:
            tracks.setdefault(event.track_id, []).append(event)

        for track_events in tracks.values():
            track_events.sort(key=lambda e: e.start_time)
            for left, right in zip(track_events, track_events[1:]):
                left_end = int(round(left.end_time * fps))
                right_start = int(round(right.start_time * fps))
                if left_end < right_start:
                    continue
                coarse_left_end = coarse[left.event_id][1]
                coarse_right_start = coarse[right.event_id][0]
                boundary = int((coarse_left_end + coarse_right_start) // 2)
                left_end = min(left_end, boundary)
                right_start = max(right_start, boundary + 1)
                left.end_time = max(left.start_time + (1.0 / fps), left_end / fps)
                right.start_time = min(right.end_time - (1.0 / fps), right_start / fps)
