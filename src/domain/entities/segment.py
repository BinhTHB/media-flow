from dataclasses import dataclass, field
from typing import Any, Literal

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
    confidence_available: bool = True
    bbox: dict[str, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

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
        if not isinstance(self.confidence_available, bool):
            raise ValidationError("Segment.confidence_available must be a boolean.")
        if self.bbox is not None:
            if not isinstance(self.bbox, dict):
                raise ValidationError("Segment.bbox must be a dictionary.")
            for key in ("left", "top", "right", "bottom"):
                if not isinstance(self.bbox.get(key), (int, float)):
                    raise ValidationError(f"Segment.bbox.{key} must be numeric.")
            if self.bbox["left"] >= self.bbox["right"] or self.bbox["top"] >= self.bbox["bottom"]:
                raise ValidationError("Segment.bbox must have positive area.")
        if not isinstance(self.metadata, dict):
            raise ValidationError("Segment.metadata must be a dictionary.")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "translated_text": self.translated_text,
            "source": self.source,
            "confidence": self.confidence,
            "confidence_available": self.confidence_available,
            "bbox": self.bbox,
            "metadata": self.metadata,
        }