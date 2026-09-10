"""
BBox Core Model

Represents a pixel-space rectangle inside a video frame.
Pure Python, zero third-party dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float
    frame_width: int
    frame_height: int

    def __post_init__(self):
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError(f"Frame dimensions must be positive, got {self.frame_width}x{self.frame_height}")

        coordinates = tuple(float(value) for value in (self.x1, self.y1, self.x2, self.y2))
        x1, y1, x2, y2 = coordinates
        if not (0.0 <= x1 < x2 <= float(self.frame_width)):
            raise ValueError(
                f"BBox x coordinates must satisfy 0 <= x1 < x2 <= {self.frame_width}, "
                f"got {x1}..{x2}"
            )
        if not (0.0 <= y1 < y2 <= float(self.frame_height)):
            raise ValueError(
                f"BBox y coordinates must satisfy 0 <= y1 < y2 <= {self.frame_height}, "
                f"got {y1}..{y2}"
            )

        object.__setattr__(self, "x1", x1)
        object.__setattr__(self, "y1", y1)
        object.__setattr__(self, "x2", x2)
        object.__setattr__(self, "y2", y2)

    @classmethod
    def clamped(
        cls,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        frame_width: int,
        frame_height: int,
        *,
        minimum_size: float = 1.0,
    ) -> "BBox":
        """Build a valid box by explicitly clamping raw pipeline geometry."""
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError(f"Frame dimensions must be positive, got {frame_width}x{frame_height}")
        minimum = float(minimum_size)
        if minimum <= 0 or minimum > frame_width or minimum > frame_height:
            raise ValueError("minimum_size must be positive and fit within the frame")

        left = max(0.0, min(float(frame_width), float(x1)))
        top = max(0.0, min(float(frame_height), float(y1)))
        right = max(0.0, min(float(frame_width), float(x2)))
        bottom = max(0.0, min(float(frame_height), float(y2)))
        if right <= left:
            left = min(left, float(frame_width) - minimum)
            right = left + minimum
        if bottom <= top:
            top = min(top, float(frame_height) - minimum)
            bottom = top + minimum
        return cls(left, top, right, bottom, frame_width, frame_height)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def normalized(self) -> Tuple[float, float, float, float]:
        return (
            self.x1 / self.frame_width,
            self.y1 / self.frame_height,
            self.x2 / self.frame_width,
            self.y2 / self.frame_height,
        )

    def xyxy_int(self) -> Tuple[int, int, int, int]:
        return int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2))

    def iou(self, other: "BBox") -> float:
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        union_area = self.area + other.area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "normalized": [round(v, 6) for v in self.normalized],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BBox":
        return cls(
            x1=float(data["x1"]),
            y1=float(data["y1"]),
            x2=float(data["x2"]),
            y2=float(data["y2"]),
            frame_width=int(data["frame_width"]),
            frame_height=int(data["frame_height"]),
        )
