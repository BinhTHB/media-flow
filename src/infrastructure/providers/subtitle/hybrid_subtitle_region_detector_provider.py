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
                 device: str = "cpu") -> None:
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
            median_bbox = track.get("median_bbox_xyxy") or []
            result = classify_track_by_position(
                median_bbox, H, rep_ids, track["track_id"], self._bottom_ratio_threshold
            )
            if result.get("uncertain"):
                continue
            decisions = result.get("decisions", [])
            if not decisions:
                continue
            if not any(d.get("is_subtitle") for d in decisions):
                continue

            conf = _clamp01(track.get("mean_detector_confidence", 0.0))
            if conf < self._min_confidence:
                continue

            # temporal union bbox for region
            union = track.get("temporal_union_bbox_xyxy") or median_bbox
            if not union:
                continue
            x1, y1, x2, y2 = union

            first_idx = track["groups"][0]["frame_index"]
            last_idx = track["groups"][-1]["frame_index"]
            start_time = frames[first_idx].timestamp
            end_time = frames[last_idx].timestamp
            if end_time < start_time:
                end_time = start_time
            # Pad by half the frame gap: frames are sampled at the middle of
            # each interval, so the region must also cover the span before the
            # first frame and after the last frame.
            if len(frames) >= 2:
                frame_gap = (frames[-1].timestamp - frames[0].timestamp) / (len(frames) - 1)
            else:
                frame_gap = 0.0
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
