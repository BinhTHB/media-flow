"""YOLO burn-in detector (Ultralytics-backed).

This is the concrete detector implementation used in production. It:

* loads the model exactly once;
* auto-selects CUDA or CPU (with manual override);
* enables FP16 on CUDA only;
* runs inference for a single frame or a batch;
* converts Ultralytics results into the internal :class:`Detection` schema
  (no Ultralytics objects leak out);
* runs per-class post-processing and temporal stabilization;
* logs model path, device, class names, and load time;
* supports explicit resource release.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ultralytics import YOLO

from src.infrastructure.detection.base import BurnInDetector
from src.infrastructure.detection.config import DetectionConfig
from src.infrastructure.detection.paths import resolve_yolo_weights
from src.infrastructure.detection.postprocess import (
    postprocess_frame,
)
from src.infrastructure.detection.schemas import (
    Detection,
    DetectionResult,
    verify_class_mapping,
)
from src.infrastructure.detection.temporal import TemporalStabilizer

logger = logging.getLogger(__name__)

#: Shared detector instances keyed by resolved weights path, so multiple
#: pipeline components that use the same model share one loaded model.
_DETECTOR_CACHE: Dict[str, "YoloBurnInDetector"] = {}


class YoloBurnInDetector(BurnInDetector):
    """Ultralytics YOLO implementation of :class:`BurnInDetector`."""

    def __init__(
        self,
        config: Optional[DetectionConfig] = None,
        weights: Optional[str] = None,
        device: Optional[str] = None,
        imgsz: Optional[int] = None,
        iou: Optional[float] = None,
        max_det: Optional[int] = None,
        half: Optional[str] = None,
        agnostic_nms: Optional[bool] = None,
        verbose: Optional[bool] = None,
        confidence: Optional[Dict[str, float]] = None,
        temporal: Optional[Any] = None,
        *,
        share_model: bool = True,
    ) -> None:
        self.config = config or DetectionConfig()
        if weights is not None:
            self.config.weights = weights
        if device is not None:
            self.config.device = device
        if imgsz is not None:
            self.config.imgsz = imgsz
        if iou is not None:
            self.config.iou = iou
        if max_det is not None:
            self.config.max_det = max_det
        if half is not None:
            self.config.half = half
        if agnostic_nms is not None:
            self.config.agnostic_nms = agnostic_nms
        if verbose is not None:
            self.config.verbose = verbose
        if confidence is not None:
            self.config.confidence = {**self.config.confidence, **confidence}

        weights_path = resolve_yolo_weights(
            config_value=self.config.weights,
            env_value=self._env_weights(),
        )

        self.weights_path: Path = weights_path
        self._device = self.config.device_resolved()
        self._half = self.config.use_half()
        self._verify_device()

        load_start = time.monotonic()
        self._model = YOLO(str(weights_path))
        self._load_time_s = time.monotonic() - load_start

        self._class_names: Dict[int, str] = dict(self._model.names)
        # Fail fast if the class mapping is not the expected 4-class mapping.
        verify_class_mapping([self._class_names[i] for i in sorted(self._class_names)])

        if self._device == "cuda":
            self._model.to(self._device)
            self._half = self.config.use_half()

        self.temporal = temporal if isinstance(temporal, TemporalStabilizer) else TemporalStabilizer(
            enabled=self.config.temporal.enabled,
            max_gap_frames=self.config.temporal.max_gap_frames,
            min_track_frames=self.config.temporal.min_track_frames,
            iou_match_threshold=self.config.temporal.iou_match_threshold,
            box_smoothing_alpha=self.config.temporal.box_smoothing_alpha,
        )

        logger.info(
            "YOLO burn-in detector loaded: path=%s device=%s imgsz=%s classes=%s "
            "half=%s load_time=%.3fs",
            self.weights_path,
            self._device,
            self.config.imgsz,
            self._class_names,
            self._half,
            self._load_time_s,
        )

    @staticmethod
    def _env_weights() -> Optional[str]:
        import os

        return os.environ.get("MEDIA_FLOW_YOLO_WEIGHTS")

    def _verify_device(self) -> None:
        if self._device in {"cuda", "gpu"}:
            try:
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "YOLO detector requested CUDA but CUDA is unavailable. "
                        "Set device='cpu' or 'auto' to run on CPU."
                    )
            except ImportError as exc:  # pragma: no cover - torch always present
                raise RuntimeError(
                    "YOLO detector requested CUDA but PyTorch is not installed."
                ) from exc

    # ------------------------------------------------------------------
    @property
    def device(self) -> str:
        return self._device

    def num_classes(self) -> int:
        return len(self._class_names)

    def get_class_names(self) -> Dict[int, str]:
        return dict(self._class_names)

    # ------------------------------------------------------------------
    def _result_to_detections(
        self, result: Any, frame_index: int, timestamp_seconds: float
    ) -> List[Detection]:
        import torch

        detections: List[Detection] = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        H, W = (int(result.orig_shape[0]), int(result.orig_shape[1]))
        boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes.xyxy, "cpu") else result.boxes.xyxy
        confs = result.boxes.conf
        if hasattr(confs, "cpu"):
            confs = confs.cpu().numpy()
        cls = result.boxes.cls
        if hasattr(cls, "cpu"):
            cls = cls.cpu().numpy()
        for xyxy, conf, class_id in zip(boxes, confs, cls):
            class_id = int(class_id)
            name = self._class_names.get(class_id, "unknown")
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=name,
                    confidence=float(conf),
                    bbox_xyxy=[float(v) for v in xyxy],
                    bbox_normalized=[
                        float(xyxy[0]) / W,
                        float(xyxy[1]) / H,
                        float(xyxy[2]) / W,
                        float(xyxy[3]) / H,
                    ],
                    frame_index=frame_index,
                    timestamp_seconds=float(timestamp_seconds),
                    frame_width=int(W),
                    frame_height=int(H),
                )
            )
        return detections

    def detect_frame(
        self, bgr_image, *, frame_index: int = 0, timestamp_seconds: float = 0.0
    ) -> DetectionResult:
        result = self._run_predict([bgr_image])[0]
        raw = self._result_to_detections(result, frame_index, timestamp_seconds)
        post = postprocess_frame(
            raw,
            self.config.confidence,
            width=result.orig_shape[1],
            height=result.orig_shape[0],
        )
        stabilized = self.temporal.stabilize(frame_index, post)
        res = DetectionResult(
            frame_index=frame_index,
            timestamp_seconds=float(timestamp_seconds),
            frame_width=int(result.orig_shape[1]),
            frame_height=int(result.orig_shape[0]),
            detections=[d._with_bbox(*d.bbox_xyxy) for d in stabilized],
        )
        res.bgr_image = bgr_image
        return res

    def detect_batch(
        self,
        bgr_images,
        *,
        frame_index_offset: int = 0,
        timestamps_seconds: Optional[List[float]] = None,
    ) -> List[DetectionResult]:
        images = list(bgr_images)
        if not images:
            return []
        raw_results = self._run_predict(images)
        out: List[DetectionResult] = []
        for offset, result in enumerate(raw_results):
            fi = frame_index_offset + offset
            ts = (
                float(timestamps_seconds[offset])
                if timestamps_seconds and offset < len(timestamps_seconds)
                else float(fi)
            )
            raw = self._result_to_detections(result, fi, ts)
            post = postprocess_frame(
                raw,
                self.config.confidence,
                width=result.orig_shape[1],
                height=result.orig_shape[0],
            )
            stabilized = self.temporal.stabilize(fi, post)
            res = DetectionResult(
                frame_index=fi,
                timestamp_seconds=float(ts),
                frame_width=int(result.orig_shape[1]),
                frame_height=int(result.orig_shape[0]),
                detections=[d._with_bbox(*d.bbox_xyxy) for d in stabilized],
            )
            res.bgr_image = images[offset]
            out.append(res)
        return out

    def _run_predict(self, images) -> Any:
        import torch

        kwargs: Dict[str, Any] = {
            "imgsz": self.config.imgsz,
            "conf": min(self.config.confidence.values()) if self.config.confidence else 0.15,
            "iou": self.config.iou,
            "max_det": self.config.max_det,
            "agnostic_nms": self.config.agnostic_nms,
            "verbose": self.config.verbose,
        }
        if self._half:
            # Prefer the unified `quantize` (FP16) introduced in newer Ultralytics;
            # fall back to the deprecated `half` flag on older versions.
            try:
                kwargs["quantize"] = 16
            except Exception:  # pragma: no cover - fallback only
                kwargs["half"] = True
        with torch.inference_mode():
            return self._model.predict(images, **kwargs)

    def release(self) -> None:
        try:
            import gc
            import torch

            if self._device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            del self._model
            gc.collect()
        except Exception:
            pass
        _DETECTOR_CACHE.pop(str(self.weights_path), None)


def get_yolo_detector(
    config: Optional[DetectionConfig] = None,
    *,
    weights: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs: Any,
) -> YoloBurnInDetector:
    """Return a shared :class:`YoloBurnInDetector`, creating it once per weights path.

    Multiple pipeline components that use the same model share a single loaded
    model, avoiding redundant memory and load time.
    """
    cfg = config or DetectionConfig()
    if weights is not None:
        cfg.weights = weights
    weights_path = resolve_yolo_weights(
        config_value=cfg.weights, env_value=YoloBurnInDetector._env_weights()
    )
    key = str(weights_path)
    if key in _DETECTOR_CACHE:
        return _DETECTOR_CACHE[key]
    detector = YoloBurnInDetector(
        config=cfg, weights=weights, device=device, **kwargs
    )
    _DETECTOR_CACHE[key] = detector
    return detector
