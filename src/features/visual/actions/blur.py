"""
Blur Action

Applies Gaussian pixel blur over a frame region.
"""

from __future__ import annotations

from typing import Any, Optional

from src.core.models.bbox import BBox


class BlurAction:
    def __init__(self, strength: float = 25.0, padding_px: int = 4):
        self.strength = strength
        self.padding_px = padding_px

    def execute(self, frame: Any, bbox: BBox, padding_px: Optional[int] = None) -> Any:
        import cv2

        pad = self.padding_px if padding_px is None else padding_px
        x1, y1, x2, y2 = bbox.xyxy_int()
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(bbox.frame_width, x2 + pad)
        y2 = min(bbox.frame_height, y2 + pad)

        if x2 <= x1 or y2 <= y1:
            return frame

        fh = bbox.frame_height or frame.shape[0]
        # Adaptive blur strength: ensure text strokes are completely dissolved
        base_strength = max(float(self.strength), fh * 0.040)
        ksize = int(round(base_strength))
        if ksize % 2 == 0:
            ksize += 1
        ksize = max(15, ksize)

        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            b1 = cv2.GaussianBlur(roi, (ksize, ksize), 0)
            blurred_roi = cv2.GaussianBlur(b1, (ksize, ksize), 0)
            frame[y1:y2, x1:x2] = blurred_roi

        return frame
