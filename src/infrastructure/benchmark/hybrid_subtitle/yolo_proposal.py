from __future__ import annotations
import json, logging, os
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
from ultralytics import YOLO
# bbox_iou not needed here
logger = logging.getLogger(__name__)


class YoloTextProposalProvider:
    def __init__(self, weights_path: str,
                 conf: float = 0.05, iou: float = 0.70,
                 imgsz: int = 1024, device: str = "cuda"):
        self._model = YOLO(weights_path)
        self._conf = conf
        self._iou = iou
        self._imgsz = imgsz
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self._device = device

    def detect(self, video_path: str) -> List[Dict[str, Any]]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("Cannot open video: " + str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        detections = []
        results = self._model.predict(
            video_path, conf=self._conf, iou=self._iou,
            imgsz=self._imgsz, stream=True, device=self._device, verbose=False)
        frame_counts: Dict[int, int] = {}
        for fi, r in enumerate(results):
            if fi >= total_frames:
                break
            H, W = r.orig_shape
            ts = fi / video_fps * 1000.0
            
            for box in r.boxes:
                frame_counts[fi] = frame_counts.get(fi, 0) + 1
                ci = frame_counts[fi]
                xyxy = [float(x) for x in box.xyxy[0].tolist()]
                norm = [xyxy[0]/W, xyxy[1]/H, xyxy[2]/W, xyxy[3]/H]
                detections.append({
                    "frame_index": fi,
                    "timestamp_ms": round(ts, 1),
                    "candidate_id": f"F{fi:05d}_C{ci:02d}",
                    "bbox_xyxy": xyxy,
                    "bbox_normalized": [round(v, 6) for v in norm],
                    "confidence": round(float(box.conf[0]), 4),
                    "class": "text",
                })
        return detections

    def detect_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Giong detect() nhung chay tren danh sach anh tinh.
        frame_index = vi tri anh trong list dau vao."""
        if not image_paths:
            return []
        detections: List[Dict[str, Any]] = []
        results = self._model.predict(
            [str(p) for p in image_paths], conf=self._conf, iou=self._iou,
            imgsz=self._imgsz, stream=True, device=self._device, verbose=False)
        frame_counts: Dict[int, int] = {}
        for fi, r in enumerate(results):
            H, W = r.orig_shape
            for box in r.boxes:
                frame_counts[fi] = frame_counts.get(fi, 0) + 1
                ci = frame_counts[fi]
                xyxy = [float(x) for x in box.xyxy[0].tolist()]
                norm = [xyxy[0]/W, xyxy[1]/H, xyxy[2]/W, xyxy[3]/H]
                detections.append({
                    "frame_index": fi,
                    "timestamp_ms": round(fi * 1000.0, 1),
                    "candidate_id": f"F{fi:05d}_C{ci:02d}",
                    "bbox_xyxy": xyxy,
                    "bbox_normalized": [round(v, 6) for v in norm],
                    "confidence": round(float(box.conf[0]), 4),
                    "class": "text",
                })
        return detections
