"""
Class-Aware Spatial-Temporal Tracker

Tracks detections across frames while preserving class_id and class_name.
Does NOT enforce bottom-band position restrictions (works for top, center, bottom detections).
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from src.domain.entities.canonical_detection import (
    CanonicalDetection,
    ClassAwareTrack,
    DetectionGroup,
)


def _box_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _union_box(boxes: Sequence[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    if not boxes:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


class ClassAwareTracker:
    def __init__(
        self,
        horizontal_gap_ratio: float = 2.0,
        track_iou_threshold: float = 0.20,
        max_gap_frames: int = 5,
        video_width: int = 1920,
        video_height: int = 1080,
    ):
        self.horizontal_gap_ratio = horizontal_gap_ratio
        self.track_iou_threshold = track_iou_threshold
        self.max_gap_frames = max_gap_frames
        self.video_width = video_width
        self.video_height = video_height

    def track(self, detections: List[CanonicalDetection]) -> List[ClassAwareTrack]:
        if not detections:
            return []

        # 1. Group detections by class_id
        by_class: Dict[int, List[CanonicalDetection]] = {}
        for det in detections:
            by_class.setdefault(det.class_id, []).append(det)

        tracks: List[ClassAwareTrack] = []
        track_counter = 1

        for class_id, class_dets in sorted(by_class.items()):
            # Group by frame_index
            by_frame: Dict[int, List[CanonicalDetection]] = {}
            for d in class_dets:
                by_frame.setdefault(d.frame_index, []).append(d)

            # Group frame candidates horizontally per class
            frame_groups: List[DetectionGroup] = []
            for frame_index in sorted(by_frame):
                dets_on_frame = by_frame[frame_index]
                groups_on_frame = self._group_frame_candidates(dets_on_frame, frame_index)
                frame_groups.extend(groups_on_frame)

            # Link frame groups across time into tracks for this class
            class_tracks = self._link_groups_to_tracks(frame_groups, track_counter)
            track_counter += len(class_tracks)
            tracks.extend(class_tracks)

        return tracks

    def _group_frame_candidates(
        self, detections: List[CanonicalDetection], frame_index: int
    ) -> List[DetectionGroup]:
        remaining = sorted(detections, key=lambda d: (d.bbox_xyxy[1], d.bbox_xyxy[0]))
        groups: List[List[CanonicalDetection]] = []

        while remaining:
            group = [remaining.pop(0)]
            changed = True
            while changed:
                changed = False
                group_box = _union_box([d.bbox_xyxy for d in group])
                group_h = group_box[3] - group_box[1]
                for item in list(remaining):
                    ix1, iy1, ix2, iy2 = item.bbox_xyxy
                    overlap = max(0.0, min(group_box[3], iy2) - max(group_box[1], iy1))
                    item_h = iy2 - iy1
                    vertical_overlap = overlap / min(group_h, item_h) if min(group_h, item_h) > 0 else 0
                    gap = max(0.0, max(group_box[0], ix1) - min(group_box[2], ix2))

                    if vertical_overlap >= 0.3 and gap <= self.horizontal_gap_ratio * max(group_h, item_h):
                        group.append(item)
                        remaining.remove(item)
                        changed = True

            groups.append(group)

        result: List[DetectionGroup] = []
        for idx, g in enumerate(groups, start=1):
            union_xyxy = _union_box([d.bbox_xyxy for d in g])
            union_norm = (
                union_xyxy[0] / self.video_width,
                union_xyxy[1] / self.video_height,
                union_xyxy[2] / self.video_width,
                union_xyxy[3] / self.video_height,
            )
            mean_conf = sum(d.confidence for d in g) / len(g)
            result.append(
                DetectionGroup(
                    group_id=f"G{frame_index:05d}_{idx:02d}",
                    class_id=g[0].class_id,
                    class_name=g[0].class_name,
                    frame_index=frame_index,
                    timestamp_sec=g[0].timestamp_sec,
                    union_bbox_xyxy=union_xyxy,
                    union_bbox_normalized=union_norm,
                    detections=g,
                    mean_confidence=mean_conf,
                )
            )
        return result

    def _link_groups_to_tracks(
        self, frame_groups: List[DetectionGroup], start_counter: int
    ) -> List[ClassAwareTrack]:
        tracks: List[ClassAwareTrack] = []
        for group in sorted(frame_groups, key=lambda g: (g.frame_index, g.group_id)):
            candidates = [
                t for t in tracks
                if group.frame_index - t.end_frame <= self.max_gap_frames
            ]
            best_track = max(
                candidates,
                key=lambda t: _box_iou(t.groups[-1].union_bbox_xyxy, group.union_bbox_xyxy),
                default=None,
            )
            if (
                best_track is not None
                and _box_iou(best_track.groups[-1].union_bbox_xyxy, group.union_bbox_xyxy)
                >= self.track_iou_threshold
            ):
                best_track.groups.append(group)
            else:
                track_id = f"T{start_counter + len(tracks):03d}_{group.class_name}"
                tracks.append(
                    ClassAwareTrack(
                        track_id=track_id,
                        class_id=group.class_id,
                        class_name=group.class_name,
                        groups=[group],
                    )
                )
        return tracks
