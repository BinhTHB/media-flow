"""Burn-in text detection infrastructure.

This package provides a clean, provider-driven detection layer for burn-in media
text (subtitle, watermark, logo, other burn-in text). It is infrastructure: it
may depend on Application ports and Domain entities, and it wraps third-party
detectors (Ultralytics YOLO) so that no Ultralytics objects leak into other
layers.

The key abstraction is :class:`~src.infrastructure.detection.base.BurnInDetector`
and its YOLO implementation :class:`~src.infrastructure.detection.yolo_detector.YoloBurnInDetector`.
"""

from src.infrastructure.detection.paths import (
    MEDIA_FLOW_YOLO_WEIGHTS_ENV,
    resolve_project_root,
    resolve_yolo_weights,
)
from src.infrastructure.detection.schemas import (
    EXPECTED_CLASS_NAMES,
    Detection,
    DetectionResult,
    class_name,
    verify_class_mapping,
)
from src.infrastructure.detection.base import BurnInDetector
from typing import Any

from src.infrastructure.detection.config import DetectionConfig, load_detection_config


def __getattr__(name: str) -> Any:
    """Load the optional Ultralytics adapter only when it is requested."""
    if name in {"YoloBurnInDetector", "get_yolo_detector"}:
        from src.infrastructure.detection.yolo_detector import (
            YoloBurnInDetector,
            get_yolo_detector,
        )

        return {
            "YoloBurnInDetector": YoloBurnInDetector,
            "get_yolo_detector": get_yolo_detector,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "MEDIA_FLOW_YOLO_WEIGHTS_ENV",
    "resolve_project_root",
    "resolve_yolo_weights",
    "EXPECTED_CLASS_NAMES",
    "Detection",
    "DetectionResult",
    "class_name",
    "verify_class_mapping",
    "BurnInDetector",
    "DetectionConfig",
    "load_detection_config",
    "YoloBurnInDetector",
    "get_yolo_detector",
]
