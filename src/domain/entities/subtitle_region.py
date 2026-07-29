from dataclasses import dataclass
from typing import Optional

from src.shared.exceptions import ValidationError


@dataclass
class SubtitleRegion:
    start_time: float
    end_time: float
    x: float
    y: float
    width: float
    height: float
    confidence: float
    # Track identifier from the text finder; mask rendering gates on track_id.
    track_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.start_time, (int, float)) or self.start_time < 0:
            raise ValidationError("start_time must be a non-negative number.")
        if not isinstance(self.end_time, (int, float)) or self.end_time < self.start_time:
            raise ValidationError(
                "end_time must be a number greater than or equal to start_time."
            )
        for name, value in (
            ("x", self.x),
            ("y", self.y),
            ("width", self.width),
            ("height", self.height),
        ):
            if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ValidationError(f"{name} must be a normalized number between 0.0 and 1.0.")
        if not isinstance(self.confidence, (int, float)) or not 0.0 <= self.confidence <= 1.0:
            raise ValidationError("confidence must be a number between 0.0 and 1.0.")

    def to_dict(self, include_provenance: bool = False) -> dict:
        data = {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
        }
        if include_provenance:
            data.update({
                "track_id": self.track_id,
            })
        return data
