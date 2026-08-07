"""
Solid Cover Action

Fills a frame region with a solid color.
"""

from __future__ import annotations

from typing import Any, Tuple

from src.core.models.bbox import BBox


def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (b, g, r)
    return (0, 0, 0)


class SolidCoverAction:
    def __init__(self, color_hex: str = "#000000", padding_px: int = 4):
        self.color_hex = color_hex
        self.padding_px = padding_px
        self.bgr = _hex_to_bgr(color_hex)

    def execute(self, frame: Any, bbox: BBox) -> Any:
        x1, y1, x2, y2 = bbox.xyxy_int()
        x1 = max(0, x1 - self.padding_px)
        y1 = max(0, y1 - self.padding_px)
        x2 = min(bbox.frame_width, x2 + self.padding_px)
        y2 = min(bbox.frame_height, y2 + self.padding_px)

        if x2 > x1 and y2 > y1:
            frame[y1:y2, x1:x2] = self.bgr
        return frame
