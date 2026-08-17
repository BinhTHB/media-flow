"""
Track Core Model

Associates same-class detections across frames into a spatial-temporal track.
Pure Python, zero third-party dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.core.models.bbox import BBox
from src.core.models.detection import Detection


@dataclass
class Track:
    track_id: str
    class_id: int
    class_name: str
    detections: List[Detection] = field(default_factory=list)

    @property
    def start_time(self) -> float:
        return self.detections[0].timestamp_seconds if self.detections else 0.0

    @property
    def end_time(self) -> float:
        return self.detections[-1].timestamp_seconds if self.detections else 0.0

    @property
    def start_frame(self) -> int:
        return self.detections[0].frame_index if self.detections else 0

    @property
    def end_frame(self) -> int:
        return self.detections[-1].frame_index if self.detections else 0

    @property
    def mean_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return sum(d.confidence for d in self.detections) / len(self.detections)

    @property
    def median_bbox(self) -> BBox:
        if not self.detections:
            raise ValueError("Cannot calculate median bbox for empty track.")
        mid = len(self.detections) // 2
        return self.detections[mid].bbox

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "start_time": round(self.start_time, 4),
            "end_time": round(self.end_time, 4),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "mean_confidence": round(self.mean_confidence, 4),
            "detections": [d.to_dict() for d in self.detections],
        }
