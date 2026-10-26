"""
Subtitle Renderer

Anchors translated text overlay to source subtitle position.
Supports adaptive font size, line wrapping, and bounded width expansion.
"""

from __future__ import annotations

from typing import Any

from src.core.models.bbox import BBox
from src.core.models.subtitle_event import SubtitleEvent
from src.features.visual.actions.text_overlay import TextOverlayAction


class SubtitleRenderer:
    def render_event(
        self,
        frame: Any,
        event: SubtitleEvent,
        bbox: BBox,
        anchor_to_bbox: bool = True,
    ) -> Any:
        if not event.translated_text:
            return frame

        overlay_action = TextOverlayAction(
            text=event.translated_text,
            stroke_width=3,
            bg_box=False,
            anchor_to_bbox=anchor_to_bbox,
        )
        return overlay_action.execute(frame, bbox)

