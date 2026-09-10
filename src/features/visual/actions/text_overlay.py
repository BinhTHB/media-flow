"""
Text Overlay Action

Renders Unicode translated subtitles with a readable modern caption panel.
The background covers both the translated text and the original subtitle region,
so residual source glyphs/blur patches are not visible around the replacement.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.core.models.bbox import BBox


def _wrap_text(text: str, max_chars_per_line: int = 30) -> str:
    if len(text) <= max_chars_per_line:
        return text
    words = text.split(" ")
    lines = []
    curr = []
    curr_len = 0
    for word in words:
        extra = len(word) + (1 if curr else 0)
        if curr and curr_len + extra > max_chars_per_line:
            lines.append(" ".join(curr))
            curr = [word]
            curr_len = len(word)
        else:
            curr.append(word)
            curr_len += extra
    if curr:
        lines.append(" ".join(curr))
    return "\n".join(lines)


def _default_font_path() -> Optional[str]:
    candidates = [
        r"C:\Windows\Fonts\seguisb.ttf",   # Segoe UI Semibold: strong Vietnamese coverage
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\bahnschrift.ttf",
    ]
    return next((path for path in candidates if os.path.isfile(path)), None)


class TextOverlayAction:
    def __init__(
        self,
        text: str,
        font_path: Optional[str] = None,
        font_size: Optional[int] = None,
        stroke_width: int = 2,
        bg_box: bool = True,
        bg_color: Tuple[int, int, int, int] = (12, 16, 22, 218),
        anchor_to_bbox: bool = False,
    ):
        self.text = text
        self.font_path = font_path
        self.font_size = font_size
        self.stroke_width = stroke_width
        self.bg_box = bg_box
        self.anchor_to_bbox = anchor_to_bbox
        self.bg_color = bg_color

    def execute(self, frame: Any, bbox: BBox) -> Any:
        import cv2

        if not self.text:
            return frame

        x1, y1, x2, y2 = bbox.xyxy_int()
        target_w = max(10, x2 - x1)
        target_h = max(10, y2 - y1)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Professional subtitle sizing is based on the frame, not the source OCR
        # box. Source boxes can be narrow fragments and previously caused absurd
        # wrapping/clipping (e.g. one Vietnamese word orphaned on a third line).
        fh = bbox.frame_height or frame.shape[0]
        fw = bbox.frame_width or frame.shape[1]
        font_size = self.font_size or max(28, min(40, int(round(fh * 0.046))))
        font_path = self.font_path or _default_font_path()
        try:
            font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

        # Wrap against the video safe area, not the width of a Chinese/English
        # source row. Keep captions to at most two balanced lines in practice.
        safe_w = int(fw * 0.74)
        if self.anchor_to_bbox:
            safe_w = min(fw - 36, max(target_w, int(fw * 0.28)))
        approx_chars = max(28, min(46, int(safe_w / max(8.0, font_size * 0.52))))
        wrapped_text = _wrap_text(self.text, max_chars_per_line=approx_chars)

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
        measure = ImageDraw.Draw(pil_img)
        text_box = measure.multiline_textbbox(
            (0, 0), wrapped_text, font=font, spacing=max(4, font_size // 7), align="center", stroke_width=self.stroke_width
        )
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]

        # Standard bottom-center subtitle safe area. Placement is independent of
        # the source row after source suppression, which avoids jitter between
        # Chinese/English stacked tracks and keeps visual hierarchy consistent.
        tx = (fw - text_w) // 2
        safe_bottom = fh - max(44, int(fh * 0.075))
        ty = safe_bottom - text_h
        if self.anchor_to_bbox:
            tx = cx - text_w // 2
            ty = cy - text_h // 2
        tx = max(18, min(fw - text_w - 18, tx))
        ty = max(18, min(fh - text_h - 18, ty))

        if self.bg_box:
            # Optional compact caption backing only; source subtitle removal is a
            # separate stage now, so this must never expand to cover the source bbox.
            text_pad_x = max(12, int(font_size * 0.30))
            text_pad_y = max(7, int(font_size * 0.18))

            bx1 = max(0, tx - text_pad_x)
            by1 = max(0, ty - text_pad_y)
            bx2 = min(fw, tx + text_w + text_pad_x)
            by2 = min(fh, ty + text_h + text_pad_y)

            if self.anchor_to_bbox:
                source_pad = max(10, int(font_size * 0.45))
                bx1 = max(0, min(bx1, x1 - source_pad))
                by1 = max(0, min(by1, y1 - source_pad))
                bx2 = min(fw, max(bx2, x2 + source_pad))
                by2 = min(fh, max(by2, y2 + source_pad))

            overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
            draw_overlay = ImageDraw.Draw(overlay)
            radius = max(8, min(18, int((by2 - by1) * 0.16)))
            if self.anchor_to_bbox:
                radius = min(radius, max(2, source_pad // 2))
            draw_overlay.rounded_rectangle(
                [bx1, by1, bx2, by2],
                radius=radius,
                fill=self.bg_color,
            )
            pil_img = Image.alpha_composite(pil_img, overlay)

        draw = ImageDraw.Draw(pil_img)
        spacing = max(4, font_size // 7)
        draw.multiline_text(
            (tx + 1, ty + 2),
            wrapped_text,
            font=font,
            fill=(0, 0, 0, 155),
            stroke_width=max(1, self.stroke_width),
            stroke_fill=(0, 0, 0, 155),
            align="center",
            spacing=spacing,
        )
        draw.multiline_text(
            (tx, ty),
            wrapped_text,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=max(3, self.stroke_width),
            stroke_fill=(0, 0, 0, 255),
            align="center",
            spacing=spacing,
        )

        return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
