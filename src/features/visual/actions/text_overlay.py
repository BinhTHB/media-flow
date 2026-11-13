"""
Text Overlay Action

Renders Unicode translated subtitles with a readable modern caption panel.
The background covers both the translated text and the original subtitle region,
so residual source glyphs/blur patches are not visible around the replacement.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.core.models.bbox import BBox


def _wrap_text(text: str, max_chars_per_line: int = 30) -> str:
    # Some models emit escaped "\n" as literal backslash-n characters; render
    # them as real line breaks instead of drawing "\n" on the frame.
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n")
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
    lines = _balance_two_lines(lines)
    return "\n".join(lines)


def _balance_two_lines(lines: List[str]) -> List[str]:
    """Re-split a greedy two-line wrap so both lines are near-equal.

    Greedy wrapping orphans the last word ("Nói năng nhẹ / nhàng"); a human
    would break where the two lines balance ("Nói năng / nhẹ nhàng").
    """
    if len(lines) != 2:
        return lines
    first_words = lines[0].split(" ")
    second_words = lines[1].split(" ")
    if len(second_words) > 1:
        best = (lines[0], lines[1])
        best_gap = abs(len(lines[0]) - len(lines[1]))
        for take in range(1, len(second_words)):
            moved = second_words[:take]
            candidate = (
                " ".join([*first_words, *moved]),
                " ".join(second_words[take:]),
            )
            gap = abs(len(candidate[0]) - len(candidate[1]))
            if gap < best_gap:
                best, best_gap = candidate, gap
        return list(best)
    if len(first_words) > 2:
        # Only the first line has words to give back; pull the tail over so
        # the orphan disappears.
        best = (lines[0], lines[1])
        best_gap = abs(len(lines[0]) - len(lines[1]))
        for give in range(len(first_words) - 1, 0, -1):
            candidate = (
                " ".join(first_words[:give]),
                " ".join([*first_words[give:], *second_words]),
            )
            gap = abs(len(candidate[0]) - len(candidate[1]))
            if gap < best_gap:
                best, best_gap = candidate, gap
        return list(best)
    return lines


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
    #: Auto-fit bounds as fractions of the base font size: long captions
    #: shrink to at most 0.72x before wrapping adds lines, roomy boxes grow
    #: short captions to at most 1.2x. Keeping the band narrow avoids the
    #: font-size jitter a per-caption free-for-all would produce.
    MIN_FONT_SCALE = 0.72
    MAX_FONT_SCALE = 1.2

    def __init__(
        self,
        text: str,
        font_path: Optional[str] = None,
        font_size: Optional[int] = None,
        stroke_width: int = 2,
        bg_box: bool = True,
        bg_color: Tuple[int, int, int, int] = (12, 16, 22, 218),
        anchor_to_bbox: bool = False,
        vertical_anchor: str = "center",
    ):
        self.text = text
        self.font_path = font_path
        self.font_size = font_size
        self.stroke_width = stroke_width
        self.bg_box = bg_box
        self.anchor_to_bbox = anchor_to_bbox
        self.vertical_anchor = vertical_anchor if vertical_anchor in {"center", "bottom"} else "center"
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
        base_font_size = self.font_size or max(28, min(40, int(round(fh * 0.046))))
        font_path = self.font_path or _default_font_path()

        # Wrap target: the video safe area for frame-anchored captions, the
        # region itself for box-anchored ones (the box then hugs the text).
        safe_w = int(fw * 0.74)
        if self.anchor_to_bbox:
            safe_w = max(120, int(target_w * 0.92))

        def _layout_for(font_size: int):
            try:
                font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.truetype("arial.ttf", font_size)
            except OSError:
                font = ImageFont.load_default()
            approx_chars = max(12, min(52, int(safe_w / max(8.0, font_size * 0.52))))
            wrapped_text = _wrap_text(self.text, max_chars_per_line=approx_chars)
            measure = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
            text_box = measure.multiline_textbbox(
                (0, 0), wrapped_text, font=font, spacing=max(4, font_size // 7), align="center", stroke_width=self.stroke_width
            )
            return font, wrapped_text, text_box[2] - text_box[0], text_box[3] - text_box[1]

        font_size = base_font_size
        font, wrapped_text, text_w, text_h = _layout_for(font_size)

        if self.anchor_to_bbox:
            # Flexible font: fit the caption inside its own box, shrinking
            # long captions and gently growing short ones, within the scale
            # band so consecutive captions stay visually consistent.
            pad_x = max(12, int(font_size * 0.30))
            pad_y = max(7, int(font_size * 0.20))
            avail_w = max(40, target_w - 2 * pad_x)
            avail_h = max(30, target_h - 2 * pad_y)
            if text_w > avail_w or text_h > avail_h:
                scale = max(self.MIN_FONT_SCALE, min(avail_w / max(1, text_w), avail_h / max(1, text_h), 1.0))
            elif text_w < avail_w * 0.55 and text_h < avail_h * 0.55:
                scale = min(self.MAX_FONT_SCALE, avail_w / max(1, text_w), avail_h / max(1, text_h))
            else:
                scale = 1.0
            if abs(scale - 1.0) > 0.02:
                font_size = max(18, int(round(font_size * scale)))
                font, wrapped_text, text_w, text_h = _layout_for(font_size)

        # Standard bottom-center subtitle safe area. Placement is independent of
        # the source row after source suppression, which avoids jitter between
        # Chinese/English stacked tracks and keeps visual hierarchy consistent.
        tx = (fw - text_w) // 2
        safe_bottom = fh - max(44, int(fh * 0.075))
        ty = safe_bottom - text_h
        if self.anchor_to_bbox:
            tx = cx - text_w // 2
            if self.vertical_anchor == "bottom":
                # Fixed baseline: the caption sits at a constant offset from
                # the box's (band-normalized) bottom edge, so 1-line and
                # 2-line captions share one baseline instead of floating
                # with their box height.
                ty = y2 - max(8, int(font_size * 0.24)) - text_h
            else:
                ty = cy - text_h // 2
        tx = max(18, min(fw - text_w - 18, tx))
        ty = max(18, min(fh - text_h - 18, ty))

        # Draw the caption on a patch the size of the line, then composite
        # that patch. Converting the whole frame to PIL on every frame was
        # most of the reup render time once OCR was no longer in the way.
        stroke = max(3, self.stroke_width)
        margin = stroke + 4
        left = tx - margin
        top = ty - margin
        right = tx + text_w + margin
        bottom = ty + text_h + margin + 2
        bg_rect = None
        if self.bg_box:
            text_pad_x = max(12, int(font_size * 0.30))
            text_pad_y = max(7, int(font_size * 0.18))
            bx1 = tx - text_pad_x
            by1 = ty - text_pad_y
            bx2 = tx + text_w + text_pad_x
            by2 = ty + text_h + text_pad_y
            if self.anchor_to_bbox:
                source_w = max(1, x2 - x1)
                text_span = text_w + 2 * text_pad_x
                # Cover the source row when it is a real caption box. A
                # cinema-wide recovery crop (80% of the frame) is not: expanding
                # onto it paints a black bar across the scene.
                cover_source = source_w <= max(text_span * 2.2, int(fw * 0.62))
                if cover_source:
                    source_pad = 4
                    bx1 = min(bx1, x1 - source_pad)
                    by1 = min(by1, y1 - source_pad)
                    bx2 = max(bx2, x2 + source_pad)
                    by2 = max(by2, y2 + source_pad)
                    bx1 = min(bx1, x1)
                    by1 = min(by1, y1)
                    bx2 = max(bx2, x2)
                    by2 = max(by2, y2)
            bg_rect = (bx1, by1, bx2, by2)
            left = min(left, bx1)
            top = min(top, by1)
            right = max(right, bx2)
            bottom = max(bottom, by2)

        left = max(0, int(left))
        top = max(0, int(top))
        right = min(int(fw), int(right))
        bottom = min(int(fh), int(bottom))
        if right <= left or bottom <= top:
            return frame

        patch = Image.new("RGBA", (right - left, bottom - top), (0, 0, 0, 0))
        draw = ImageDraw.Draw(patch)
        if bg_rect is not None:
            bx1, by1, bx2, by2 = bg_rect
            px1 = max(0, bx1 - left)
            py1 = max(0, by1 - top)
            px2 = min(right - left, bx2 - left)
            py2 = min(bottom - top, by2 - top)
            radius = max(10, min(18, int((py2 - py1) * 0.22)))
            radius = min(radius, max(0, (px2 - px1) // 2), max(0, (py2 - py1) // 2))
            draw.rounded_rectangle(
                [px1, py1, px2, py2],
                radius=radius,
                fill=self.bg_color,
            )
        spacing = max(4, font_size // 7)
        local_tx = tx - left
        local_ty = ty - top
        draw.multiline_text(
            (local_tx + 1, local_ty + 2),
            wrapped_text,
            font=font,
            fill=(0, 0, 0, 155),
            stroke_width=max(1, self.stroke_width),
            stroke_fill=(0, 0, 0, 155),
            align="center",
            spacing=spacing,
        )
        draw.multiline_text(
            (local_tx, local_ty),
            wrapped_text,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=stroke,
            stroke_fill=(0, 0, 0, 255),
            align="center",
            spacing=spacing,
        )

        roi = frame[top:bottom, left:right]
        if roi.size == 0:
            return frame
        base = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)).convert("RGBA")
        merged = Image.alpha_composite(base, patch)
        frame[top:bottom, left:right] = cv2.cvtColor(
            np.array(merged.convert("RGB")), cv2.COLOR_RGB2BGR
        )
        return frame
