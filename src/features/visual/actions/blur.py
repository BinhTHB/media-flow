"""
Blur Action

Applies Gaussian pixel blur over a frame region.
"""

from __future__ import annotations

from typing import Any, Tuple

from src.core.models.bbox import BBox


class BlurAction:
    def __init__(self, strength: float = 25.0, padding_px: int = 4):
        self.strength = strength
        self.padding_px = padding_px

    def execute(self, frame: Any, bbox: BBox) -> Any:
        import cv2

        x1, y1, x2, y2 = bbox.xyxy_int()
        x1 = max(0, x1 - self.padding_px)
        y1 = max(0, y1 - self.padding_px)
        x2 = min(bbox.frame_width, x2 + self.padding_px)
        y2 = min(bbox.frame_height, y2 + self.padding_px)

        if x2 <= x1 or y2 <= y1:
            return frame

        ksize = int(round(self.strength))
        if ksize % 2 == 0:
            ksize += 1
        ksize = max(3, ksize)

        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            blurred_roi = cv2.GaussianBlur(roi, (ksize, ksize), 0)
            frame[y1:y2, x1:x2] = blurred_roi

        return frame
