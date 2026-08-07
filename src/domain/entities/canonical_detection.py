"""
Canonical Detection Domain Entities

Defines the single unified detection schema and class-aware track entities
that preserve class_id, class_name, confidence, bbox, and timestamps across the entire pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

EXPECTED_CLASS_MAPPING = {
    0: "subtitle",
    1: "watermark",
    2: "logo",
    3: "other_burn_in_text",
}


@dataclass
class CanonicalDetection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]
    bbox_normalized: Tuple[float, float, float, float]
    frame_index: int
    timestamp_sec: float
    candidate_id: str = ""
    track_id: str = ""

    def __post_init__(self):
        if self.class_id in EXPECTED_CLASS_MAPPING and not self.class_name:
            self.class_name = EXPECTED_CLASS_MAPPING[self.class_id]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox_xyxy": [round(float(v), 2) for v in self.bbox_xyxy],
            "bbox_normalized": [round(float(v), 6) for v in self.bbox_normalized],
            "frame_index": self.frame_index,
            "timestamp_sec": round(float(self.timestamp_sec), 4),
            "candidate_id": self.candidate_id,
            "track_id": self.track_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalDetection":
        bbox_xyxy = tuple(data["bbox_xyxy"])
        bbox_normalized = tuple(data["bbox_normalized"])
        return cls(
            class_id=int(data["class_id"]),
            class_name=str(data.get("class_name") or EXPECTED_CLASS_MAPPING.get(data["class_id"], "unknown")),
            confidence=float(data["confidence"]),
            bbox_xyxy=bbox_xyxy,  # type: ignore
            bbox_normalized=bbox_normalized,  # type: ignore
            frame_index=int(data["frame_index"]),
            timestamp_sec=float(data["timestamp_sec"]),
            candidate_id=str(data.get("candidate_id", "")),
            track_id=str(data.get("track_id", "")),
        )


@dataclass
class DetectionGroup:
    group_id: str
    class_id: int
    class_name: str
    frame_index: int
    timestamp_sec: float
    union_bbox_xyxy: Tuple[float, float, float, float]
    union_bbox_normalized: Tuple[float, float, float, float]
    detections: List[CanonicalDetection] = field(default_factory=list)
    mean_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
            "union_bbox_xyxy": [round(float(v), 2) for v in self.union_bbox_xyxy],
            "union_bbox_normalized": [round(float(v), 6) for v in self.union_bbox_normalized],
            "mean_confidence": round(float(self.mean_confidence), 4),
            "detections": [d.to_dict() for d in self.detections],
        }


@dataclass
class ClassAwareTrack:
    track_id: str
    class_id: int
    class_name: str
    groups: List[DetectionGroup] = field(default_factory=list)

    @property
    def start_frame(self) -> int:
        return self.groups[0].frame_index if self.groups else 0

    @property
    def end_frame(self) -> int:
        return self.groups[-1].frame_index if self.groups else 0

    @property
    def start_time(self) -> float:
        return self.groups[0].timestamp_sec if self.groups else 0.0

    @property
    def end_time(self) -> float:
        return self.groups[-1].timestamp_sec if self.groups else 0.0

    @property
    def mean_confidence(self) -> float:
        if not self.groups:
            return 0.0
        return sum(g.mean_confidence for g in self.groups) / len(self.groups)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": round(self.start_time, 4),
            "end_time": round(self.end_time, 4),
            "frame_count": len(self.groups),
            "mean_confidence": round(self.mean_confidence, 4),
            "groups": [g.to_dict() for g in self.groups],
        }
