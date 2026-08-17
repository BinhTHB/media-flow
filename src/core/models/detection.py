"""
Detection Core Model

Single frame object detection model.
Pure Python, zero third-party dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

from src.core.models.bbox import BBox

CANONICAL_CLASS_NAMES = {
    0: "subtitle",
    1: "watermark",
    2: "logo",
    3: "other_burn_in_text",
}


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: BBox
    frame_index: int
    timestamp_seconds: float
    candidate_id: str = ""

    def __post_init__(self):
        if self.class_id in CANONICAL_CLASS_NAMES and not self.class_name:
            self.class_name = CANONICAL_CLASS_NAMES[self.class_id]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": self.bbox.to_dict(),
            "frame_index": self.frame_index,
            "timestamp_seconds": round(float(self.timestamp_seconds), 4),
            "candidate_id": self.candidate_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Detection":
        return cls(
            class_id=int(data["class_id"]),
            class_name=str(data.get("class_name") or CANONICAL_CLASS_NAMES.get(data["class_id"], "unknown")),
            confidence=float(data["confidence"]),
            bbox=BBox.from_dict(data["bbox"]),
            frame_index=int(data["frame_index"]),
            timestamp_seconds=float(data["timestamp_seconds"]),
            candidate_id=str(data.get("candidate_id", "")),
        )
