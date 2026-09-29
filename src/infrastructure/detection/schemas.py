"""Detection data schema.

Defines the internal representation of a single detection and a frame result.
The schema is framework-agnostic (plain dataclasses) so Ultralytics objects never
leak into other pipeline layers. All bounding boxes are clamped to the frame
bounds and each detection is JSON-serializable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.core.models.detection import CANONICAL_CLASS_NAMES

#: Canonical class-id -> class-name mapping, aliased from the core model so
#: there is exactly one source of truth for burn-in class semantics.
EXPECTED_CLASS_NAMES: Dict[int, str] = dict(CANONICAL_CLASS_NAMES)


class ClassMappingError(ValueError):
    """Raised when the loaded model's class mapping diverges from expectations."""


def class_name(class_id: int) -> str:
    """Return the canonical name for a class id, or ``unknown`` if missing."""
    return EXPECTED_CLASS_NAMES.get(int(class_id), "unknown")


def verify_class_mapping(names: Sequence[str]) -> Dict[int, str]:
    """Verify a model's class names against the expected mapping.

    The given ``names`` is a sequence where index == class id. This fails fast
    (raises :class:`ClassMappingError`) if the mapping does not match the
    expected 4-class burn-in mapping, instead of silently running with wrong
    labels. Returns a normalized ``{class_id: class_name}`` dict when valid.
    """
    names = [str(n) for n in names]
    if len(names) != len(EXPECTED_CLASS_NAMES):
        raise ClassMappingError(
            f"Model has {len(names)} classes but {len(EXPECTED_CLASS_NAMES)} were "
            f"expected. Model classes: {names}. Expected: {list(EXPECTED_CLASS_NAMES.values())}."
        )
    expected = list(EXPECTED_CLASS_NAMES.values())
    if names != expected:
        raise ClassMappingError(
            f"Model class names {names!r} do not match expected {expected!r}. "
            "Refusing to run with incorrect burn-in labels."
        )
    normalized = {int(i): str(name) for i, name in enumerate(names)}
    return normalized


@dataclass
class Detection:
    """A single detected burn-in region on one frame.

    ``bbox_xyxy`` is always clamped to the frame bounds. Normalized coordinates
    are relative to the frame width/height.
    """

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: List[float]  # x1, y1, x2, y2
    bbox_normalized: List[float]
    frame_index: int
    timestamp_seconds: float
    frame_width: int
    frame_height: int

    # Optional post-processed grouping info (filled by postprocess/grouping).
    region_id: Optional[str] = None
    group_id: Optional[str] = None
    track_id: Optional[str] = None

    def clamp(self) -> "Detection":
        """Return a copy with ``bbox_xyxy`` clamped into frame bounds."""
        x1, y1, x2, y2 = self.bbox_xyxy
        x1 = max(0.0, min(float(self.frame_width), x1))
        y1 = max(0.0, min(float(self.frame_height), y1))
        x2 = max(0.0, min(float(self.frame_width), x2))
        y2 = max(0.0, min(float(self.frame_height), y2))
        c = self._with_bbox(x1, y1, x2, y2)
        return c

    def _with_bbox(self, x1: float, y1: float, x2: float, y2: float) -> "Detection":
        from src.infrastructure.detection.postprocess import compute_normalized

        return Detection(
            class_id=self.class_id,
            class_name=self.class_name,
            confidence=self.confidence,
            bbox_xyxy=[x1, y1, x2, y2],
            bbox_normalized=compute_normalized(
                [x1, y1, x2, y2], self.frame_width, self.frame_height
            ),
            frame_index=self.frame_index,
            timestamp_seconds=self.timestamp_seconds,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            region_id=self.region_id,
            group_id=self.group_id,
            track_id=self.track_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json_compatible(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict (all primitives, floats rounded)."""
        return {
            "class_id": int(self.class_id),
            "class_name": str(self.class_name),
            "confidence": round(float(self.confidence), 6),
            "bbox_xyxy": [round(float(v), 6) for v in self.bbox_xyxy],
            "bbox_normalized": [round(float(v), 6) for v in self.bbox_normalized],
            "frame_index": int(self.frame_index),
            "timestamp_seconds": round(float(self.timestamp_seconds), 6),
            "frame_width": int(self.frame_width),
            "frame_height": int(self.frame_height),
            "region_id": self.region_id,
            "group_id": self.group_id,
            "track_id": self.track_id,
        }


@dataclass
class DetectionResult:
    """Detections for a single frame."""

    frame_index: int
    timestamp_seconds: float
    frame_width: int
    frame_height: int
    detections: List[Detection] = field(default_factory=list)

    @property
    def bgr_image(self) -> Optional[Any]:
        """BGR frame associated with this result (BGR to match OpenCV/Ultralytics)."""
        return getattr(self, "_bgr_image", None)

    @bgr_image.setter
    def bgr_image(self, image: Any) -> None:
        self._bgr_image = image

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "detections": [d.to_json_compatible() for d in self.detections],
        }
