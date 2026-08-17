"""
SpeechSegment Core Model

Represents an ASR audio speech segment.
Pure Python, zero third-party dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SpeechSegment:
    segment_id: str
    start_time: float
    end_time: float
    source_text: str
    source_language: str = "auto"
    speaker_id: Optional[str] = None
    translated_text: str = ""

    def __post_init__(self):
        if self.end_time <= self.start_time:
            self.end_time = self.start_time + 0.1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "start_time": round(self.start_time, 4),
            "end_time": round(self.end_time, 4),
            "source_text": self.source_text,
            "source_language": self.source_language,
            "speaker_id": self.speaker_id,
            "translated_text": self.translated_text,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpeechSegment":
        return cls(
            segment_id=str(data["segment_id"]),
            start_time=float(data["start_time"]),
            end_time=float(data["end_time"]),
            source_text=str(data.get("source_text", "")),
            source_language=str(data.get("source_language", "auto")),
            speaker_id=data.get("speaker_id"),
            translated_text=str(data.get("translated_text", "")),
        )
