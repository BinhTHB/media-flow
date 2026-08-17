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
        # Clamp coordinates to frame boundary
        cx1 = max(0.0, min(float(self.frame_width), float(self.x1)))
        cy1 = max(0.0, min(float(self.frame_height), float(self.y1)))
        cx2 = max(0.0, min(float(self.frame_width), float(self.x2)))
        cy2 = max(0.0, min(float(self.frame_height), float(self.y2)))

        if cx2 <= cx1:
            cx2 = min(float(self.frame_width), cx1 + 1.0)
        if cy2 <= cy1:
            cy2 = min(float(self.frame_height), cy1 + 1.0)

        object.__setattr__(self, "x1", cx1)
        object.__setattr__(self, "y1", cy1)
        object.__setattr__(self, "x2", cx2)
        object.__setattr__(self, "y2", cy2)

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
