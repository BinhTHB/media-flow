from typing import Sequence
from src.application.services.dynamic_ass_event_service import DynamicAssEvent
from src.domain.entities.subtitle_style import SubtitleStyle
from src.application.services.ass_generator import (
    generate_ass_header,
    _format_time,
    _escape_text,
)

class DynamicAssDocumentService:
    @staticmethod
    def generate_dynamic_replacement(
        events: Sequence[DynamicAssEvent],
        style: SubtitleStyle,
        play_res_x: int = 1920,
        play_res_y: int = 1080,
    ) -> str:
        header = generate_ass_header(style, play_res_x, play_res_y)
        ass_events = []
        for ev in events:
            start = _format_time(ev.start)
            end = _format_time(ev.end)
            escaped_text = _escape_text(ev.text)
            mask = ev.cover_bbox
            if mask:
                left = round(mask.get('left', 0.0))
                top = round(mask.get('top', 0.0))
                width = round(mask.get('right', 0.0) - mask.get('left', 0.0))
                height = round(mask.get('bottom', 0.0) - mask.get('top', 0.0))
                cover_line = (
                    f"Dialogue: 0,{start},{end},Default,,0,0,0,,"
                    rf"{{\an7\pos({left},{top})\p1\1c&H000000&\1a&H00&}}"
                    f"m 0 0 l {width} 0 l {width} {height} l 0 {height}"
                    r"{\p0}"
                )
                ass_events.append(cover_line)
            tags = [r"\an5"]
            if ev.font_size != style.font_size:
                tags.append(rf"\fs{ev.font_size}")
            tags.append(rf"\pos({round(ev.position_x)},{round(ev.position_y)})")
            tag_str = "{" + "".join(tags) + "}"
            text_line = (
                f"Dialogue: 1,{start},{end},Default,,0,0,0,,"
                f"{tag_str}{escaped_text}"
            )
            ass_events.append(text_line)
        return header + "\n" + "\n".join(ass_events)