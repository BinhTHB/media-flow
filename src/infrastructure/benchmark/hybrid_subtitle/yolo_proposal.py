from __future__ import annotations

import json, logging, os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from ultralytics import YOLO

from src.application.dto.proposal_bundle import DecodedFrame, ProposalBundle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# YoloTextProposalProvider — now supports both legacy detect() / detect_images()
# and the new single-decode detect_video()
# ---------------------------------------------------------------------------

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
        """Decode video once, materialize frames from YOLO results.

        Fails loud on:
        - Unreadable video (cv2.VideoCapture.open returns False)
        - Blank/gray frames (pixel std < min_pixel_std)
        - Resolution mismatch (decoded shape != metadata shape)
        - Zero frames produced

        Parameters
        ----------
        video_path : str
            Path to input video.
        stride : int
            Frame stride: emit 1 every `stride` frames (1 = every frame).
        output_dir : str | None
            If set, dump each decoded frame to disk inside this directory.
        min_pixel_std : float
            Minimum pixel standard deviation to consider a frame non-blank.
            Default 1.0 rejects pure gray frames (std=0).

        Returns
        -------
        ProposalBundle
            Contains all decoded frames with YOLO detections, provenance metadata.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video for decode: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if total_frames <= 0:
            raise RuntimeError(
                f"Video reports 0 frames (total_frames={total_frames}). "
                "Check video integrity or codec support."
            )
        if video_w <= 0 or video_h <= 0:
            raise RuntimeError(
                f"Video reports invalid resolution: {video_w}x{video_h}. "
                "Check video metadata."
            )

        bundle = ProposalBundle(
            video_path=str(video_path),
            fps=video_fps,
            stride=stride,
            video_width=video_w,
            video_height=video_h,
            total_frames=total_frames,
        )

        results = self._model.predict(
            video_path, conf=self._conf, iou=self._iou,
            imgsz=self._imgsz, stream=True, device=self._device, verbose=False)

        frame_counts: Dict[int, int] = {}

        for fi, r in enumerate(results):
            if fi >= total_frames:
                break
            if fi % stride != 0:
                # Skip frames outside the stride pattern
                continue

            H, W = r.orig_shape
            ts = fi / video_fps

            # Resolution sanity check
            if H != video_h or W != video_w:
                raise RuntimeError(
                    f"Frame {fi} resolution mismatch: got {W}x{H}, "
                    f"expected {video_w}x{video_h} from metadata. "
                    "Video may have been misreported or has variable resolution."
                )

            # Blank frame validation
            orig_img = r.orig_img
            if orig_img is None:
                raise RuntimeError(
                    f"YOLO returned no image data at frame_index={fi}. "
                    "Video decode may have failed at this position."
                )
            pixel_std = float(orig_img.std())
            if pixel_std < min_pixel_std:
                raise RuntimeError(
                    f"Blank/gray frame detected at frame_index={fi} "
                    f"(std={pixel_std:.4f} < {min_pixel_std}). "
                    "Video decode may have failed or the codec is unsupported. "
                    f"Resolution: {W}x{H}."
                )

            # Write frame to disk if requested
            img_path: str | None = None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                img_path = os.path.join(output_dir, f"frame_{fi:06d}.jpg")
                cv2.imwrite(img_path, orig_img)

            # Collect detections for this frame
            frame_detections: List[Dict[str, Any]] = []
            for box in r.boxes:
                frame_counts[fi] = frame_counts.get(fi, 0) + 1
                ci = frame_counts[fi]
                xyxy = [float(x) for x in box.xyxy[0].tolist()]
                norm = [xyxy[0] / W, xyxy[1] / H, xyxy[2] / W, xyxy[3] / H]
                frame_detections.append({
                    "frame_index": fi,
                    "timestamp_ms": round(ts * 1000.0, 1),
                    "candidate_id": f"F{fi:05d}_C{ci:02d}",
                    "bbox_xyxy": xyxy,
                    "bbox_normalized": [round(v, 6) for v in norm],
                    "confidence": round(float(box.conf[0]), 4),
                    "class": "text",
                })

            bundle.add_frame(
                frame_index=fi,
                timestamp_sec=round(ts, 4),
                image=orig_img,
                image_path=img_path,
                detections=frame_detections,
            )

        if not bundle.frames:
            raise RuntimeError(
                f"YOLO produced 0 frames from video {video_path} "
                f"(total_frames={total_frames}, stride={stride}). "
                "Video decode likely failed silently."
            )

        return bundle

    def detect(self, video_path: str) -> List[Dict[str, Any]]:
        """Legacy: decode video and return detections list only.

        Kept for backward compatibility and for cases that don't need frames.
        For new code prefer detect_video() which preserves frame data.
        """
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
        """Legacy: detect on pre-extracted image list.

        Kept for unit tests and legacy paths only. New code should use
        detect_video() which processes video in one pass without separate
        frame extraction.
        """
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
