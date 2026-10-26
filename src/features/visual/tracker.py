"""
Class-Aware Spatial-Temporal Tracker

Tracks detections across frames while preserving class_id and class_name (0=subtitle, 1=watermark, 2=logo, 3=other_burn_in_text).
Pure Python, zero third-party dependencies.
No bottom-band position restrictions (works for top, center, bottom detections).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from src.core.models.bbox import BBox
from src.core.models.detection import Detection
from src.core.models.track import Track


def _box_iou(a: BBox, b: BBox) -> float:
    return a.iou(b)


def _union_bbox(boxes: Sequence[BBox]) -> BBox:
    if not boxes:
        raise ValueError("At least one BBox is required for union.")
    return BBox(
        x1=min(b.x1 for b in boxes),
        y1=min(b.y1 for b in boxes),
        x2=max(b.x2 for b in boxes),
        y2=max(b.y2 for b in boxes),
        frame_width=boxes[0].frame_width,
        frame_height=boxes[0].frame_height,
    )


class ClassAwareTracker:
    def __init__(
        self,
        horizontal_gap_ratio: float = 2.0,
        track_iou_threshold: float = 0.20,
        max_gap_frames: int = 5,
    ):
        self.horizontal_gap_ratio = horizontal_gap_ratio
        self.track_iou_threshold = track_iou_threshold
        self.max_gap_frames = max_gap_frames

    def track(self, detections: List[Detection]) -> List[Track]:
        if not detections:
            return []

        # 1. Group detections by class_id
        by_class: Dict[int, List[Detection]] = {}
        for det in detections:
            by_class.setdefault(det.class_id, []).append(det)

        tracks: List[Track] = []
        track_counter = 1

        for class_id, class_dets in sorted(by_class.items()):
            # Group by frame_index
            by_frame: Dict[int, List[Detection]] = {}
            for d in class_dets:
                by_frame.setdefault(d.frame_index, []).append(d)

            # Group frame candidates horizontally per class
            frame_groups: List[List[Detection]] = []
            for frame_index in sorted(by_frame):
                dets_on_frame = by_frame[frame_index]
                groups_on_frame = self._group_frame_candidates(dets_on_frame)
                frame_groups.extend(groups_on_frame)

            # Link frame groups across time into tracks for this class
            class_tracks = self._link_groups_to_tracks(frame_groups, track_counter)
            track_counter += len(class_tracks)
            tracks.extend(class_tracks)

        # Post-process: merge broken static tracks for non-subtitle classes
        tracks = self._merge_static_tracks(tracks)
        return tracks

    def _merge_static_tracks(
        self, tracks: List[Track], max_gap_frames: int = 150, iou_thresh: float = 0.50
    ) -> List[Track]:
        sub = [t for t in tracks if t.class_name == "subtitle"]
        non_sub = [t for t in tracks if t.class_name != "subtitle"]
        if not non_sub:
            return tracks

        sorted_tracks = sorted(non_sub, key=lambda t: t.start_frame)
        merged: List[Track] = []

        for t in sorted_tracks:
            matched = False
            t_rep_box = t.detections[len(t.detections) // 2].bbox
            for m in merged:
                m_rep_box = m.detections[len(m.detections) // 2].bbox
                iou = _box_iou(t_rep_box, m_rep_box)
                gap = t.start_frame - m.end_frame
                if iou >= iou_thresh and 0 <= gap <= max_gap_frames:
                    m.detections.extend(t.detections)
                    m.detections.sort(key=lambda d: d.frame_index)
                    matched = True
                    break
            if not matched:
                merged.append(
                    Track(
                        track_id=t.track_id,
                        class_id=t.class_id,
                        class_name=t.class_name,
                        detections=list(t.detections),
                    )
                )

        return sub + merged

    def _group_frame_candidates(self, detections: List[Detection]) -> List[List[Detection]]:
        remaining = sorted(detections, key=lambda d: (d.bbox.y1, d.bbox.x1))
        groups: List[List[Detection]] = []

        while remaining:
            group = [remaining.pop(0)]
            changed = True
            while changed:
                changed = False
                group_box = _union_bbox([d.bbox for d in group])
                group_h = group_box.height
                for item in list(remaining):
                    overlap = max(0.0, min(group_box.y2, item.bbox.y2) - max(group_box.y1, item.bbox.y1))
                    item_h = item.bbox.height
                    vertical_overlap = overlap / min(group_h, item_h) if min(group_h, item_h) > 0 else 0
                    gap = max(0.0, max(group_box.x1, item.bbox.x1) - min(group_box.x2, item.bbox.x2))

                    if vertical_overlap >= 0.3 and gap <= self.horizontal_gap_ratio * max(group_h, item_h):
                        group.append(item)
                        remaining.remove(item)
                        changed = True

            groups.append(group)

        return groups

    def _link_groups_to_tracks(
        self, frame_groups: List[List[Detection]], start_counter: int
    ) -> List[Track]:
        tracks: List[Track] = []

        for group in sorted(frame_groups, key=lambda g: (g[0].frame_index, g[0].candidate_id)):
            group_box = _union_bbox([d.bbox for d in group])
            group_frame = group[0].frame_index

            candidates = [
                t for t in tracks
                if group_frame - t.end_frame <= self.max_gap_frames
            ]
            best_track = max(
                candidates,
                key=lambda t: _box_iou(t.detections[-1].bbox, group_box),
                default=None,
            )
            if (
                best_track is not None
                and _box_iou(best_track.detections[-1].bbox, group_box) >= self.track_iou_threshold
            ):
                best_track.detections.extend(group)
            else:
                track_id = f"T{start_counter + len(tracks):03d}_{group[0].class_name}"
                tracks.append(
                    Track(
                        track_id=track_id,
                        class_id=group[0].class_id,
                        class_name=group[0].class_name,
                        detections=list(group),
                    )
                )

        return tracks
