"""
YOLO Text Proposal Provider

Canonical single-pass video decode & detection materialization using Ultralytics YOLO.
Preserves class_id and class_name across all detections.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from ultralytics import YOLO

from src.application.dto.proposal_bundle import DecodedFrame, ProposalBundle
from src.infrastructure.detection.paths import resolve_yolo_weights
from src.infrastructure.detection.schemas import class_name

logger = logging.getLogger(__name__)

CROP_PADDING: int = 8


def _resize_720p(image: Any) -> Any:
    H, W = image.shape[:2]
    max_dim = max(H, W)
    if max_dim <= 720:
        return image.copy()
    scale = 720.0 / max_dim
    new_w = int(round(W * scale))
    new_h = int(round(H * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


class YoloTextProposalProvider:
    def __init__(
        self,
        weights_path: str = "models/yolo/best.pt",
        conf: float = 0.05,
        iou: float = 0.70,
        imgsz: int = 1024,
        device: str = "cuda",
    ):
        self._weights_path = resolve_yolo_weights(config_value=weights_path)
        self._model = YOLO(str(self._weights_path))
        self._class_names = {int(k): str(v) for k, v in dict(self._model.names).items()}
        self._conf = conf
        self._iou = iou
        self._imgsz = imgsz

        import torch
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "YOLO GPU was requested but CUDA is unavailable. "
                "Use device='cpu' only as an explicit opt-in."
            )
        self._device = device

    def detect_video(
        self,
        video_path: str,
        stride: int = 1,
        output_dir: str | None = None,
        min_pixel_std: float = 1.0,
    ) -> ProposalBundle:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video for decode: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if total_frames <= 0:
            raise RuntimeError(f"Video reports 0 frames (total_frames={total_frames}).")
        if video_w <= 0 or video_h <= 0:
            raise RuntimeError(f"Video reports invalid resolution: {video_w}x{video_h}.")

        bundle = ProposalBundle(
            video_path=str(video_path),
            fps=video_fps,
            stride=stride,
            video_width=video_w,
            video_height=video_h,
            total_frames=total_frames,
        )

        results = self._model.predict(
            video_path,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            stream=True,
            device=self._device,
            verbose=False,
        )

        frame_counts: Dict[int, int] = {}

        for fi, r in enumerate(results):
            if fi >= total_frames:
                break
            if fi % stride != 0:
                continue

            H, W = r.orig_shape
            ts = fi / video_fps

            if H != video_h or W != video_w:
                raise RuntimeError(
                    f"Frame {fi} resolution mismatch: got {W}x{H}, expected {video_w}x{video_h}."
                )

            orig_img = r.orig_img
            if orig_img is None:
                raise RuntimeError(f"YOLO returned no image data at frame_index={fi}.")

            pixel_std = float(orig_img.std())
            if pixel_std < min_pixel_std:
                raise RuntimeError(
                    f"Blank/gray frame detected at frame_index={fi} (std={pixel_std:.4f} < {min_pixel_std})."
                )

            img_path: str | None = None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                img_path = os.path.join(output_dir, f"frame_{fi:06d}.jpg")
                cv2.imwrite(img_path, orig_img)

            image_720p = _resize_720p(orig_img)

            frame_detections: List[Dict[str, Any]] = []
            frame_crops: list[Any] = []

            for box in r.boxes:
                frame_counts[fi] = frame_counts.get(fi, 0) + 1
                ci = frame_counts[fi]
                xyxy = [float(x) for x in box.xyxy[0].tolist()]
                norm = [xyxy[0] / W, xyxy[1] / H, xyxy[2] / W, xyxy[3] / H]
                class_id = int(box.cls[0]) if hasattr(box, "cls") and len(box.cls) else 0
                class_names = getattr(self, "_class_names", None) or {}
                name = class_names.get(class_id, class_name(class_id))

                frame_detections.append(
                    {
                        "frame_index": fi,
                        "timestamp_ms": round(ts * 1000.0, 1),
                        "candidate_id": f"F{fi:05d}_C{ci:02d}",
                        "bbox_xyxy": xyxy,
                        "bbox_normalized": [round(v, 6) for v in norm],
                        "confidence": round(float(box.conf[0]), 4),
                        "class_id": class_id,
                        "class_name": name,
                        "class": name,  # Preserves actual class_name
                    }
                )

                x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
                px1 = max(0, x1 - CROP_PADDING)
                py1 = max(0, y1 - CROP_PADDING)
                px2 = min(W, x2 + CROP_PADDING)
                py2 = min(H, y2 + CROP_PADDING)
                crop = orig_img[py1:py2, px1:px2].copy()
                frame_crops.append(crop)

            bundle.add_frame(
                frame_index=fi,
                timestamp_sec=round(ts, 4),
                image=orig_img,
                image_path=img_path,
                detections=frame_detections,
            )
            bundle.frames[-1].image_720p = image_720p
            bundle.frames[-1].crops = frame_crops

        if not bundle.frames:
            raise RuntimeError(f"YOLO produced 0 frames from video {video_path}.")

        return bundle

    def detect(self, video_path: str) -> List[Dict[str, Any]]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("Cannot open video: " + str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        detections = []
        results = self._model.predict(
            video_path,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            stream=True,
            device=self._device,
            verbose=False,
        )
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
                norm = [xyxy[0] / W, xyxy[1] / H, xyxy[2] / W, xyxy[3] / H]
                class_id = int(box.cls[0]) if hasattr(box, "cls") and len(box.cls) else 0
                class_names = getattr(self, "_class_names", None) or {}
                name = class_names.get(class_id, class_name(class_id))
                detections.append(
                    {
                        "frame_index": fi,
                        "timestamp_ms": round(ts, 1),
                        "candidate_id": f"F{fi:05d}_C{ci:02d}",
                        "bbox_xyxy": xyxy,
                        "bbox_normalized": [round(v, 6) for v in norm],
                        "confidence": round(float(box.conf[0]), 4),
                        "class_id": class_id,
                        "class_name": name,
                        "class": name,
                    }
                )
        return detections

    def detect_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        if not image_paths:
            return []
        detections: List[Dict[str, Any]] = []
        results = self._model.predict(
            [str(p) for p in image_paths],
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            stream=True,
            device=self._device,
            verbose=False,
        )
        frame_counts: Dict[int, int] = {}
        for fi, r in enumerate(results):
            H, W = r.orig_shape
            for box in r.boxes:
                frame_counts[fi] = frame_counts.get(fi, 0) + 1
                ci = frame_counts[fi]
                xyxy = [float(x) for x in box.xyxy[0].tolist()]
                norm = [xyxy[0] / W, xyxy[1] / H, xyxy[2] / W, xyxy[3] / H]
                class_id = int(box.cls[0]) if hasattr(box, "cls") and len(box.cls) else 0
                class_names = getattr(self, "_class_names", None) or {}
                name = class_names.get(class_id, class_name(class_id))
                detections.append(
                    {
                        "frame_index": fi,
                        "timestamp_ms": round(fi * 1000.0, 1),
                        "candidate_id": f"F{fi:05d}_C{ci:02d}",
                        "bbox_xyxy": xyxy,
                        "bbox_normalized": [round(v, 6) for v in norm],
                        "confidence": round(float(box.conf[0]), 4),
                        "class_id": class_id,
                        "class_name": name,
                        "class": name,
                    }
                )
        return detections
