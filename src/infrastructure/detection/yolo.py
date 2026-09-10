"""
YOLO Detector Infrastructure Adapter

Implements DetectorPort using Ultralytics YOLO models/yolo/best.pt.
Converts raw predictions into core Detection & BBox objects.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import List

from src.core.contracts import DetectorPort
from src.core.models.bbox import BBox
from src.core.models.detection import Detection, CANONICAL_CLASS_NAMES

logger = logging.getLogger(__name__)


class YoloDetectorAdapter(DetectorPort):
    def __init__(
        self,
        weights_path: str = "models/yolo/best.pt",
        conf: float = 0.05,
        iou: float = 0.70,
        imgsz: int = 1024,
        device: str = "cuda",
        subtitle_rescue_conf: float | None = 0.01,
        subtitle_rescue_max_gap_seconds: float = 3.0,
        subtitle_rescue_min_support_ratio: float = 0.25,
    ):
        from src.infrastructure.detection.yolo_proposal_provider import YoloTextProposalProvider

        self._base_conf = float(conf)
        self._subtitle_rescue_conf = (
            None if subtitle_rescue_conf is None else float(subtitle_rescue_conf)
        )
        self._subtitle_rescue_max_gap_seconds = float(subtitle_rescue_max_gap_seconds)
        self._subtitle_rescue_min_support_ratio = float(subtitle_rescue_min_support_ratio)
        inference_conf = self._base_conf
        if self._subtitle_rescue_conf is not None:
            inference_conf = min(inference_conf, self._subtitle_rescue_conf)
        self.provider = YoloTextProposalProvider(
            weights_path=weights_path,
            conf=inference_conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
        )

    @staticmethod
    def _same_subtitle_lane(a: BBox, b: BBox) -> bool:
        overlap = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
        if overlap / max(1.0, min(a.width, b.width)) < 0.45:
            return False
        gap = max(0.0, max(a.y1, b.y1) - min(a.y2, b.y2))
        return gap <= max(4.0, min(a.height, b.height) * 0.70)

    def _filter_and_rescue(self, detections: List[Detection], fps: float) -> List[Detection]:
        """Keep normal detections and bridge dense weak subtitle evidence only.

        Low-confidence inference is intentionally class-gated and temporal: weak
        subtitle boxes are accepted only inside a short gap bracketed by strong
        subtitle boxes in the same lane, with support on enough frames. This
        increases recall without lowering the runtime threshold for watermark,
        logo, or other text classes.
        """
        base_conf = getattr(self, "_base_conf", None)
        rescue_conf = getattr(self, "_subtitle_rescue_conf", None)
        if base_conf is None or rescue_conf is None or rescue_conf >= base_conf:
            return detections

        kept = [d for d in detections if d.confidence >= base_conf]
        weak_subtitles = [
            d for d in detections
            if d.class_name == "subtitle" and rescue_conf <= d.confidence < base_conf
        ]
        strong_by_frame: dict[int, list[Detection]] = defaultdict(list)
        weak_by_frame: dict[int, list[Detection]] = defaultdict(list)
        for det in kept:
            if det.class_name == "subtitle":
                strong_by_frame[det.frame_index].append(det)
        for det in weak_subtitles:
            weak_by_frame[det.frame_index].append(det)
        strong_frames = sorted(strong_by_frame)
        if len(strong_frames) < 2 or not weak_by_frame:
            return kept

        max_gap_frames = max(1, int(round(
            getattr(self, "_subtitle_rescue_max_gap_seconds", 3.0) * max(1.0, fps)
        )))
        min_support_ratio = max(0.0, min(1.0, getattr(
            self, "_subtitle_rescue_min_support_ratio", 0.25
        )))
        rescued_ids: set[int] = set()

        for left_frame, right_frame in zip(strong_frames, strong_frames[1:]):
            gap_len = right_frame - left_frame - 1
            if gap_len <= 0 or gap_len > max_gap_frames:
                continue
            lane_pairs = [
                (left, right)
                for left in strong_by_frame[left_frame]
                for right in strong_by_frame[right_frame]
                if self._same_subtitle_lane(left.bbox, right.bbox)
            ]
            if not lane_pairs:
                continue
            candidates = []
            for frame_index in range(left_frame + 1, right_frame):
                for weak in weak_by_frame.get(frame_index, []):
                    if any(
                        self._same_subtitle_lane(weak.bbox, left.bbox)
                        and self._same_subtitle_lane(weak.bbox, right.bbox)
                        for left, right in lane_pairs
                    ):
                        candidates.append(weak)
            supported_frames = len({item.frame_index for item in candidates})
            required_frames = max(3, int(round(gap_len * min_support_ratio)))
            if supported_frames < required_frames:
                continue
            rescued_ids.update(id(item) for item in candidates)

        return sorted(
            [*kept, *(d for d in weak_subtitles if id(d) in rescued_ids)],
            key=lambda d: (d.frame_index, -d.confidence, d.class_id),
        )

    def detect_video(self, video_path: str, stride: int = 1) -> List[Detection]:
        # The adapter only needs normalized detections; do not retain every full-res
        # frame/crop in RAM for long 1080p videos.
        bundle = self.provider.detect_video(video_path, stride=stride, retain_images=False)
        detections: List[Detection] = []

        w = bundle.video_width
        h = bundle.video_height

        for frame in bundle.frames:
            for det in frame.detections:
                cid = int(det.get("class_id", 0))
                cname = str(det.get("class_name") or CANONICAL_CLASS_NAMES.get(cid, "subtitle"))
                xyxy = det["bbox_xyxy"]

                bbox = BBox(
                    x1=xyxy[0],
                    y1=xyxy[1],
                    x2=xyxy[2],
                    y2=xyxy[3],
                    frame_width=w,
                    frame_height=h,
                )

                detections.append(
                    Detection(
                        class_id=cid,
                        class_name=cname,
                        confidence=float(det.get("confidence", 0.0)),
                        bbox=bbox,
                        frame_index=frame.frame_index,
                        timestamp_seconds=frame.timestamp_sec,
                        candidate_id=det.get("candidate_id", ""),
                    )
                )

        return self._filter_and_rescue(
            detections, float(getattr(bundle, "fps", 30.0) or 30.0)
        )
