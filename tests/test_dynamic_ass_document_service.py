import pytest
from src.application.services.dynamic_ass_event_service import DynamicAssEvent
from src.application.services.dynamic_ass_document_service import DynamicAssDocumentService
from src.domain.entities.subtitle_style import SubtitleStyle

def _event(interval_id="interval_1", start=1.0, end=2.0, font_size=24) -> DynamicAssEvent:
    return DynamicAssEvent(
        interval_id=interval_id,
        start=start,
        end=end,
        text="Xin chào Việt Nam",
        lines=("Xin chào Việt Nam",),
        font_size=font_size,
        position_x=100.0,
        position_y=200.0,
        cover_bbox={"left": 50.0, "top": 150.0, "right": 150.0, "bottom": 250.0},
        replacement_status="REPLACED",
    )

def test_generate_dynamic_replacement_creates_cover_and_text_dialogues() -> None:
    style = SubtitleStyle(font_family="Arial", font_size=24)
    event = _event()
    ass_content = DynamicAssDocumentService.generate_dynamic_replacement(
        [event], style, play_res_x=720, play_res_y=1280
    )
    
    assert "[Script Info]" in ass_content
    assert "PlayResX: 720" in ass_content
    assert "PlayResY: 1280" in ass_content
    assert "Style: Default,Arial" in ass_content
    
    # Check Dialogue lines
    lines = ass_content.splitlines()
    dialogue_lines = [l for l in lines if l.startswith("Dialogue:")]
    assert len(dialogue_lines) == 2
    
    # Cover line: Dialogue: 0,... (layer 0)
    cover_line = dialogue_lines[0]
    assert cover_line.startswith("Dialogue: 0,")
    # check start and end formatting: 1.0 -> 0:00:01.00, 2.0 -> 0:00:02.00
    assert "0:00:01.00" in cover_line
    assert "0:00:02.00" in cover_line
    assert "\\an7" in cover_line
    assert "\\pos(50,150)" in cover_line
    assert "\\p1" in cover_line
    assert "m 0 0 l 100 0 l 100 100 l 0 100" in cover_line
    
    # Text line: Dialogue: 1,... (layer 1)
    text_line = dialogue_lines[1]
    assert text_line.startswith("Dialogue: 1,")
    assert "0:00:01.00" in text_line
    assert "0:00:02.00" in text_line
    assert "\\an5" in text_line
    assert "\\pos(100,200)" in text_line
    assert "Xin chào Việt Nam" in text_line
    assert "\\fs" not in text_line # since event.font_size == style.font_size (24)

def test_generate_dynamic_replacement_overrides_font_size() -> None:
    style = SubtitleStyle(font_size=24)
    event = _event(font_size=18)
    ass_content = DynamicAssDocumentService.generate_dynamic_replacement(
        [event], style, play_res_x=720, play_res_y=1280
    )
    
    lines = ass_content.splitlines()
    text_line = [l for l in lines if l.startswith("Dialogue: 1,")][0]
    assert "\\fs18" in text_line

def test_generate_dynamic_replacement_two_lines_ass_line_break() -> None:
    style = SubtitleStyle(font_size=24)
    event = DynamicAssEvent(
        interval_id="interval_1",
        start=1.0,
        end=2.0,
        text="Xin chào\\NViệt Nam",
        lines=("Xin chào", "Việt Nam"),
        font_size=24,
        position_x=100.0,
        position_y=200.0,
        cover_bbox={"left": 50.0, "top": 150.0, "right": 150.0, "bottom": 250.0},
        replacement_status="REPLACED",
    )
    ass_content = DynamicAssDocumentService.generate_dynamic_replacement(
        [event], style, play_res_x=720, play_res_y=1280
    )
    assert "Xin chào\\NViệt Nam" in ass_content

def test_generate_dynamic_replacement_keeps_vietnamese_unicode() -> None:
    style = SubtitleStyle()
    event = DynamicAssEvent(
        interval_id="interval_1",
        start=1.0,
        end=2.0,
        text="Tiếng Việt thân yêu",
        lines=("Tiếng Việt thân yêu",),
        font_size=24,
        position_x=100.0,
        position_y=200.0,
        cover_bbox={"left": 50.0, "top": 150.0, "right": 150.0, "bottom": 250.0},
        replacement_status="REPLACED",
    )
    ass_content = DynamicAssDocumentService.generate_dynamic_replacement(
        [event], style
    )
    assert "Tiếng Việt thân yêu" in ass_content