"""
Subtitle Renderer

Anchors translated text overlay to source subtitle position.
Supports adaptive font size, line wrapping, and bounded width expansion.
"""

from __future__ import annotations

from typing import Any, List, Tuple

from src.core.models.bbox import BBox
from src.core.models.subtitle_event import SubtitleEvent
from src.features.visual.actions.text_overlay import TextOverlayAction


class SubtitleRenderer:
    def render_event(self, frame: Any, event: SubtitleEvent, bbox: BBox) -> Any:
        if not event.translated_text:
            return frame

        overlay_action = TextOverlayAction(
            text=event.translated_text,
            stroke_width=2,
        )
        return overlay_action.execute(frame, bbox)
