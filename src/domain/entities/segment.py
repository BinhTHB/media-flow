from dataclasses import dataclass
from typing import Literal

from src.shared.exceptions import ValidationError

SegmentSource = Literal["speech", "ocr"]

@dataclass
class Segment:
    id: str
    start: float
    end: float
    text: str
    translated_text: str = ""
    source: SegmentSource = "speech"
    confidence: float = 0.0

    def __post_init__(self):
        if not self.id:
            raise ValidationError("Segment.id cannot be empty.")
        if self.start < 0:
            raise ValidationError("Segment.start must be non-negative.")
        if self.end < self.start:
            raise ValidationError("Segment.end must be greater than or equal to start.")
        if self.source not in ("speech", "ocr"):
            raise ValidationError("Segment.source must be either 'speech' or 'ocr'.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValidationError("Segment.confidence must be between 0.0 and 1.0.")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "translated_text": self.translated_text,
            "source": self.source,
            "confidence": self.confidence,
        }