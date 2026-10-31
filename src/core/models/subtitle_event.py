"""
SubtitleEvent Core Model

Represents a single semantic subtitle text event interval.
A spatial track may contain multiple SubtitleEvents if captions change over time.
Pure Python, zero third-party dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.core.models.bbox import BBox


@dataclass
class SubtitleEvent:
    event_id: str
    track_id: str
    start_time: float
    end_time: float
    anchor_bbox: BBox
    bbox_trajectory: List[BBox] = field(default_factory=list)
    bbox_frame_indices: List[int] = field(default_factory=list)
    crop_paths: List[str] = field(default_factory=list)
    source_text: str = ""
    translated_text: str = ""
    split_reason: str = "initial"
    replacement_status: str = "SKIPPED"  # READY | OCR_EMPTY | OCR_FAILED | TRANSLATION_EMPTY | TRANSLATION_FAILED | SOURCE_IS_TARGET | TRANSLATION_UNCHANGED | SKIPPED
    blur_applied: bool = False
    overlay_applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "track_id": self.track_id,
            "start_time": round(self.start_time, 4),
            "end_time": round(self.end_time, 4),
            "duration": round(max(0.0, self.end_time - self.start_time), 4),
            "split_reason": self.split_reason,
            "replacement_status": self.replacement_status,
            "blur_applied": self.blur_applied,
            "overlay_applied": self.overlay_applied,
            "anchor_bbox": self.anchor_bbox.to_dict(),
            "bbox_trajectory": [b.to_dict() for b in self.bbox_trajectory],
            "bbox_frame_indices": list(self.bbox_frame_indices),
            "crop_paths": self.crop_paths,
            "source_text": self.source_text,
            "translated_text": self.translated_text,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubtitleEvent":
        return cls(
            event_id=data["event_id"],
            track_id=data["track_id"],
            start_time=float(data["start_time"]),
            end_time=float(data["end_time"]),
            anchor_bbox=BBox.from_dict(data["anchor_bbox"]),
            bbox_trajectory=[BBox.from_dict(b) for b in data.get("bbox_trajectory", [])],
            bbox_frame_indices=[int(v) for v in data.get("bbox_frame_indices", [])],
            crop_paths=list(data.get("crop_paths", [])),
            source_text=str(data.get("source_text", "")),
            translated_text=str(data.get("translated_text", "")),
            split_reason=str(data.get("split_reason", "initial")),
            replacement_status=str(data.get("replacement_status", "SKIPPED")),
            blur_applied=bool(data.get("blur_applied", False)),
            overlay_applied=bool(data.get("overlay_applied", False)),
        )
