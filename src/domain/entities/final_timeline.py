from dataclasses import dataclass
from typing import Any, Dict, Tuple

from src.shared.exceptions import ValidationError


@dataclass(frozen=True)
class TimelineItem:
    segment_id: str
    start: float
    end: float
    text: str

    def __post_init__(self) -> None:
        if not self.segment_id or self.start < 0 or self.end < self.start:
            raise ValidationError("Timeline item has invalid identity or timing.")

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }


@dataclass(frozen=True)
class FinalTimeline:
    timeline_id: str
    created_at: str
    source_segment_ids: Tuple[str, ...]
    narration: Tuple[TimelineItem, ...]
    display: Tuple[TimelineItem, ...]
    replacement_regions: Tuple[Dict[str, Any], ...]
    placements: Tuple[Dict[str, Any], ...]

    def __post_init__(self) -> None:
        if not self.timeline_id or not self.created_at:
            raise ValidationError("Final timeline identity is required.")
        for track in (self.narration, self.display):
            ids = [item.segment_id for item in track]
            if len(ids) != len(set(ids)):
                raise ValidationError("Timeline track segment IDs must be unique.")
        for replacement in self.replacement_regions:
            if not replacement.get("segment_id") or not replacement.get("mask_bbox"):
                raise ValidationError("Replacement timeline entries require mask_bbox.")

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "timeline_id": self.timeline_id,
            "created_at": self.created_at,
            "source_segment_ids": list(self.source_segment_ids),
            "tracks": {
                "narration": [item.to_dict() for item in self.narration],
                "display_subtitles": [item.to_dict() for item in self.display],
                "source_subtitle_replacements": list(self.replacement_regions),
                "placements": list(self.placements),
            },
        }
