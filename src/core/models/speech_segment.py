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
    spoken_text: str = ""

    def __post_init__(self):
        self.start_time = float(self.start_time)
        self.end_time = float(self.end_time)
        if self.start_time < 0:
            raise ValueError(f"SpeechSegment start_time must be non-negative, got {self.start_time}")
        if self.end_time <= self.start_time:
            raise ValueError(
                "SpeechSegment end_time must be greater than start_time, "
                f"got {self.start_time}..{self.end_time}"
            )

    @classmethod
    def normalized_timing(
        cls,
        *,
        segment_id: str,
        start_time: float,
        end_time: float,
        source_text: str,
        minimum_duration: float = 0.1,
        **kwargs: Any,
    ) -> "SpeechSegment":
        """Build a segment after explicitly normalizing provider timing."""
        start = max(0.0, float(start_time))
        duration = float(minimum_duration)
        if duration <= 0:
            raise ValueError("minimum_duration must be positive")
        end = max(start + duration, float(end_time))
        return cls(
            segment_id=segment_id,
            start_time=start,
            end_time=end,
            source_text=source_text,
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "start_time": round(self.start_time, 4),
            "end_time": round(self.end_time, 4),
            "source_text": self.source_text,
            "source_language": self.source_language,
            "speaker_id": self.speaker_id,
            "translated_text": self.translated_text,
            "spoken_text": self.spoken_text,
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
            spoken_text=str(data.get("spoken_text", "")),
        )
