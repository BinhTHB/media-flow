"""
Text Overlay Action

Renders Unicode Vietnamese text anchored over a frame region using PIL.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.core.models.bbox import BBox


def _wrap_text(text: str, max_chars_per_line: int = 32) -> str:
    if len(text) <= max_chars_per_line:
        return text
    words = text.split(" ")
    lines = []
    curr = []
    curr_len = 0
    for w in words:
        if curr and curr_len + len(w) + 1 > max_chars_per_line:
            lines.append(" ".join(curr))
            curr = [w]
            curr_len = len(w)
        else:
            curr.append(w)
            curr_len += len(w) + (1 if len(curr) > 1 else 0)
    if curr:
        lines.append(" ".join(curr))
    return "\n".join(lines)


class TextOverlayAction:
    def __init__(
        self,
        text: str,
        font_path: Optional[str] = None,
        font_size: Optional[int] = None,
        stroke_width: int = 2,
        bg_box: bool = True,
        bg_color: Tuple[int, int, int, int] = (15, 15, 15, 230),
    ):
        self.text = text
        self.font_path = font_path
        self.font_size = font_size
        self.stroke_width = stroke_width
        self.bg_box = bg_box
        self.bg_color = bg_color

    def execute(self, frame: Any, bbox: BBox) -> Any:
        import cv2

        if not self.text:
            return frame

        x1, y1, x2, y2 = bbox.xyxy_int()
        target_w = max(10, x2 - x1)
        target_h = max(10, y2 - y1)

        # Center of source bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        wrapped_text = _wrap_text(self.text, max_chars_per_line=32)

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")

        font_size = self.font_size or max(20, int(target_h * 0.65))
        try:
            font = ImageFont.truetype(self.font_path or "arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

        # Temporary draw to measure multiline text bbox
        dummy_draw = ImageDraw.Draw(pil_img)
        bbox_text = dummy_draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=4)
        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]

        # Anchor text block CENTER to source bbox CENTER
        tx = cx - text_w // 2
        ty = cy - text_h // 2

        # Clamp to frame bounds with margin
        fw = bbox.frame_width or frame.shape[1]
        fh = bbox.frame_height or frame.shape[0]

        tx = max(10, min(fw - text_w - 10, tx))
        ty = max(10, min(fh - text_h - 10, ty))

        if self.bg_box:
            pad_x = 10
            pad_y = 6
            bx1 = max(0, tx - pad_x)
            by1 = max(0, ty - pad_y)
            bx2 = min(fw, tx + text_w + pad_x)
            by2 = min(fh, ty + text_h + pad_y)

            overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
            draw_ov = ImageDraw.Draw(overlay)
            draw_ov.rectangle([bx1, by1, bx2, by2], fill=self.bg_color)
            pil_img = Image.alpha_composite(pil_img, overlay)

        draw = ImageDraw.Draw(pil_img)

        draw.multiline_text(
            (tx, ty),
            wrapped_text,
            font=font,
            fill=(255, 255, 255),
            stroke_width=self.stroke_width,
            stroke_fill=(0, 0, 0),
            align="center",
            spacing=4,
        )

        return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
