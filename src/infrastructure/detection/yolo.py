"""
YOLO Detector Infrastructure Adapter

Implements DetectorPort using Ultralytics YOLO models/yolo/best.pt.
Converts raw predictions into core Detection & BBox objects.
"""

from __future__ import annotations

import logging
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
    ):
        from src.infrastructure.detection.yolo_proposal_provider import YoloTextProposalProvider

        self.provider = YoloTextProposalProvider(
            weights_path=weights_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
        )

    def detect_video(self, video_path: str, stride: int = 1) -> List[Detection]:
        bundle = self.provider.detect_video(video_path, stride=stride)
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

        return detections
