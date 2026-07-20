from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

import cv2

from src.application.ports.subtitle_region_detector_provider import (
    SubtitleRegionDetectorProvider,
)
from src.domain.entities.frame import Frame
from src.domain.entities.subtitle_region import SubtitleRegion
from src.infrastructure.benchmark.hybrid_subtitle.core import (
    box_iou,
    group_frame_candidates,
    build_tracks,
    representative_group_ids,
)
from src.infrastructure.benchmark.hybrid_subtitle.runner import (
    classify_track_by_position,
)

logger = logging.getLogger(__name__)


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


class HybridSubtitleRegionDetectorProvider(SubtitleRegionDetectorProvider):
    """YOLO proposal -> grouping -> tracking -> heuristic vi tri.

    proposal_provider cho phep inject fake trong test (khong can best.pt).
    YOLO that chi duoc khoi tao lazy trong detect_regions.
    """

    def __init__(self, weights: str = "best.pt", conf: float = 0.05,
                 iou: float = 0.70, imgsz: int = 1024,
                 horizontal_gap_ratio: float = 2.0,
                 track_iou_threshold: float = 0.25,
                 track_max_gap_frames: int = 5,
                 bottom_ratio_threshold: float = 0.65,
                 min_confidence: float = 0.30,
                 proposal_provider: Optional[Any] = None,
                 device: str = "cpu",
                 semantic_verifier: Optional[Any] = None) -> None:
        self._weights = weights
        self._conf = conf
        self._iou = iou
        self._imgsz = imgsz
        self._h_gap_ratio = horizontal_gap_ratio
        self._track_iou_threshold = track_iou_threshold
        self._track_max_gap_frames = track_max_gap_frames
        self._bottom_ratio_threshold = bottom_ratio_threshold
        self._min_confidence = min_confidence
        self._proposal = proposal_provider  # lazy-init trong detect_regions
        self._device = device
        self._semantic_verifier = semantic_verifier

    def detect_regions(self, frames: List[Frame]) -> List[SubtitleRegion]:
        if not frames:
            return []

        # lazy-init YOLO
        if self._proposal is None:
            from src.infrastructure.benchmark.hybrid_subtitle.yolo_proposal import YoloTextProposalProvider
            self._proposal = YoloTextProposalProvider(
                weights_path=self._weights,
                conf=self._conf,
                iou=self._iou,
                imgsz=self._imgsz,
                device=self._device,
            )

        # 1) detect images
        image_paths = [f.image_path for f in frames]
        dets = self._proposal.detect_images(image_paths)

        # 2) group per frame
        by_frame: Dict[int, List[Dict]] = {}
        for d in dets:
            by_frame.setdefault(d["frame_index"], []).append(d)

        flat_groups: List[Dict] = []
        for fi in sorted(by_frame):
            groups = group_frame_candidates(by_frame[fi], self._h_gap_ratio)
            flat_groups.extend(groups)

        if not flat_groups:
            return []

        # 3) build tracks
        tracks = build_tracks(
            flat_groups,
            iou_threshold=self._track_iou_threshold,
            max_gap_frames=self._track_max_gap_frames,
        )
        if not tracks:
            return []

        # get frame dimensions
        img = cv2.imread(frames[0].image_path)
        H, W = img.shape[:2]

        regions: List[SubtitleRegion] = []
        for track in tracks:
            rep_ids = representative_group_ids(track, 5)

            # Verifier-first: when a semantic verifier is injected, it decides
            # for every track and the rule heuristic is ignored for that track.
            if self._semantic_verifier is not None:
                accepted = self._verifier_accepts_track(track, frames, H)
            else:
                accepted = self._rule_accepts_track(track, rep_ids, H)

            if not accepted:
                continue

            conf = _clamp01(track.get("mean_detector_confidence", 0.0))
            if conf < self._min_confidence:
                continue

            # Frame cadence (frames are sampled at the middle of each interval).
            if len(frames) >= 2:
                frame_gap = (frames[-1].timestamp - frames[0].timestamp) / (len(frames) - 1)
            else:
                frame_gap = 0.0

            # Detect moving track: if first and last frame bboxes diverge, emit
            # one region per frame instead of a single union that spans all frames.
            first_bbox = track["groups"][0]["union_bbox_xyxy"]
            last_bbox = track["groups"][-1]["union_bbox_xyxy"]
            moving = len(track["groups"]) >= 2 and box_iou(first_bbox, last_bbox) < 0.30

            if moving:
                for g in track["groups"]:
                    bbox = g["union_bbox_xyxy"]
                    if not bbox:
                        continue
                    x1, y1, x2, y2 = bbox
                    ts = frames[g["frame_index"]].timestamp
                    start_time = max(0.0, ts - frame_gap / 2.0)
                    end_time = ts + frame_gap / 2.0
                    try:
                        region = SubtitleRegion(
                            start_time=start_time,
                            end_time=end_time,
                            x=_clamp01(x1 / W),
                            y=_clamp01(y1 / H),
                            width=_clamp01((x2 - x1) / W),
                            height=_clamp01((y2 - y1) / H),
                            confidence=conf,
                        )
                    except Exception:
                        continue
                    regions.append(region)
                continue

            # Stable track: emit a single region spanning the full track.
            union = (
                track.get("temporal_union_bbox_xyxy")
                or track.get("median_bbox_xyxy")
                or []
            )
            if not union:
                continue
            x1, y1, x2, y2 = union

            first_idx = track["groups"][0]["frame_index"]
            last_idx = track["groups"][-1]["frame_index"]
            start_time = frames[first_idx].timestamp
            end_time = frames[last_idx].timestamp
            if end_time < start_time:
                end_time = start_time
            start_time = max(0.0, start_time - frame_gap / 2.0)
            end_time = end_time + frame_gap / 2.0

            try:
                region = SubtitleRegion(
                    start_time=start_time,
                    end_time=end_time,
                    x=_clamp01(x1 / W),
                    y=_clamp01(y1 / H),
                    width=_clamp01((x2 - x1) / W),
                    height=_clamp01((y2 - y1) / H),
                    confidence=conf,
                )
            except Exception:
                continue
            regions.append(region)

        return regions

    def _rule_accepts_track(self, track, rep_ids, H) -> bool:
        median_bbox = track.get("median_bbox_xyxy") or []
        result = classify_track_by_position(
            median_bbox, H, rep_ids, track["track_id"], self._bottom_ratio_threshold
        )
        if result.get("uncertain"):
            return False
        decisions = result.get("decisions", [])
        if not decisions:
            return False
        return any(d.get("is_subtitle") for d in decisions)

    def _verifier_accepts_track(self, track, frames, H) -> bool:
        try:
            groups = track["groups"]
            if len(groups) <= 3:
                rep_groups = groups
            else:
                rep_groups = [groups[0], groups[len(groups) // 2], groups[-1]]

            candidate_ids = [g["candidate_group_id"] for g in rep_groups]
            candidate_bboxes = [g["union_bbox_xyxy"] for g in rep_groups]
            frame_bgr = []
            for g in rep_groups:
                frame = cv2.imread(frames[g["frame_index"]].image_path)
                if frame is None:
                    continue
                frame_bgr.append(frame)

            if not frame_bgr:
                # Could not load any frame; fall back to rule-based.
                return self._rule_accepts_track(track, representative_group_ids(track, 5), H)

            class TrackWrapper:
                def __init__(self, track_id):
                    self.track_id = track_id

            # Adapt the track dict to an object with track_id for classify_track signature compatibility
            track_obj = TrackWrapper(track["track_id"])

            decisions = self._semantic_verifier.classify_track(
                track_obj, frame_bgr, candidate_ids, candidate_bboxes
            )
            decision_list = decisions.get("decisions", [])
            return any(d.get("is_subtitle") for d in decision_list)
        except Exception:
            # Fall back to rule-based on any verifier failure
            return self._rule_accepts_track(track, representative_group_ids(track, 5), H)
