from dataclasses import dataclass
from typing import Sequence

from src.application.services.dynamic_subtitle_text_layout_service import (
    DynamicSubtitleLayoutResult,
    FIT,
)
from src.application.services.dynamic_subtitle_timeline_service import (
    TimelineMappingResult,
)

@dataclass(frozen=True)
class DynamicAssEvent:
    interval_id: str
    start: float
    end: float
    text: str
    lines: tuple[str, ...]
    font_size: int
    position_x: float
    position_y: float
    cover_bbox: object
    replacement_status: str

class DynamicAssEventService:
    """Service that converts FIT layout results into ASS event specifications."""

    @staticmethod
    def build_event(
        layout: DynamicSubtitleLayoutResult,
        mapping: TimelineMappingResult,
    ) -> DynamicAssEvent | None:
        if layout.layout_status != FIT or not layout.lines:
            return None
        # join lines with ASS line break \N
        ass_text = "\\N".join(layout.lines)
        if not ass_text.strip():
            return None
        return DynamicAssEvent(
            interval_id=layout.interval_id,
            start=mapping.start,
            end=mapping.end,
            text=ass_text,
            lines=layout.lines,
            font_size=layout.font_size,
            position_x=layout.position_x,
            position_y=layout.position_y,
            cover_bbox=layout.expanded_cover_bbox,
            replacement_status="REPLACED",
        )

    @staticmethod
    def build_events(
        pairs: Sequence[tuple[DynamicSubtitleLayoutResult, TimelineMappingResult]],
    ) -> list[DynamicAssEvent]:
        events = []
        for layout, mapping in pairs:
            event = DynamicAssEventService.build_event(layout, mapping)
            if event is not None:
                events.append(event)
        return events
