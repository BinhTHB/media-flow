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
        if getattr(frame, "ndim", 0) < 2:
            return frame

        fh = bbox.frame_height or frame.shape[0]
        # Adaptive blur strength: ensure text strokes are completely dissolved
        base_strength = max(float(self.strength), fh * 0.040)
        ksize = int(round(base_strength))
        if ksize % 2 == 0:
            ksize += 1
        ksize = max(15, ksize)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return frame

        height, width = roi.shape[:2]
        # Full-resolution 40px+ kernels on 1080p watermark/subtitle plates
        # dominate visual render time. Blur a downscaled copy then scale back:
        # glyphs stay unreadable, CPU drops sharply.
        scale = 4 if min(height, width) >= 32 else 2 if min(height, width) >= 16 else 1
        if scale > 1:
            small_w = max(2, width // scale)
            small_h = max(2, height // scale)
            small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_AREA)
            small_k = max(5, ksize // scale)
            if small_k % 2 == 0:
                small_k += 1
            blurred = cv2.GaussianBlur(small, (small_k, small_k), 0)
            blurred = cv2.GaussianBlur(blurred, (small_k, small_k), 0)
            frame[y1:y2, x1:x2] = cv2.resize(blurred, (width, height), interpolation=cv2.INTER_LINEAR)
        else:
            first = cv2.GaussianBlur(roi, (ksize, ksize), 0)
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(first, (ksize, ksize), 0)

        return frame
