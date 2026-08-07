"""
Text Overlay Action

Renders Unicode Vietnamese text anchored over a frame region using PIL.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.core.models.bbox import BBox


class TextOverlayAction:
    def __init__(
        self,
        text: str,
        font_path: Optional[str] = None,
        font_size: Optional[int] = None,
        stroke_width: int = 2,
    ):
        self.text = text
        self.font_path = font_path
        self.font_size = font_size
        self.stroke_width = stroke_width

    def execute(self, frame: Any, bbox: BBox) -> Any:
        import cv2

        if not self.text:
            return frame

        x1, y1, x2, y2 = bbox.xyxy_int()
        target_w = x2 - x1
        target_h = y2 - y1

        if target_w <= 0 or target_h <= 0:
            return frame

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        font_size = self.font_size or max(16, int(target_h * 0.6))
        try:
            font = ImageFont.truetype(self.font_path or "arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

        bbox_text = draw.textbbox((0, 0), self.text, font=font)
        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]

        tx = x1 + max(0, (target_w - text_w) // 2)
        ty = y1 + max(0, (target_h - text_h) // 2)

        draw.text(
            (tx, ty),
            self.text,
            font=font,
            fill=(255, 255, 255),
            stroke_width=self.stroke_width,
            stroke_fill=(0, 0, 0),
        )

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
