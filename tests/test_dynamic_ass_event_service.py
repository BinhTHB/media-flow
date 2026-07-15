from src.application.services.dynamic_ass_event_service import (
    DynamicAssEvent,
    DynamicAssEventService,
)
from src.application.services.dynamic_subtitle_text_layout_service import (
    DynamicSubtitleLayoutResult,
    FIT,
    REVIEW_REQUIRED,
    SKIPPED,
)

def _layout(status: str, reason: str, lines=("Xin chào",), font_size=24) -> DynamicSubtitleLayoutResult:
    return DynamicSubtitleLayoutResult(
        interval_id="interval_1",
        layout_status=status,
        layout_reason=reason,
        lines=lines,
        font_size=font_size,
        position_x=55.0,
        position_y=40.0,
        text_width=64.0,
        text_height=27.6,
        source_bbox={"left": 10.0, "top": 20.0, "right": 100.0, "bottom": 60.0},
        expanded_cover_bbox={"left": 5.0, "top": 15.0, "right": 105.0, "bottom": 65.0},
    )

def _mapping(start=1.0, end=2.0):
    from src.application.services.dynamic_subtitle_timeline_service import (
        TimelineMappingResult,
    )
    from src.application.services.translation_quality_gate import (
        TranslationQualityResult,
    )
    return TimelineMappingResult(
        interval_id="interval_1",
        start=start,
        end=end,
        source_text="你好",
        target_text="Xin chào",
        source_bbox={"left": 10.0, "top": 20.0, "right": 100.0, "bottom": 60.0},
        expanded_cover_bbox={"left": 5.0, "top": 15.0, "right": 105.0, "bottom": 65.0},
        mapping_status="mapped",
        mapping_reason="timestamp_overlap",
        matched_segment_id="ocr_1",
        intersection_duration=0.8,
        interval_coverage=0.8,
        segment_coverage=0.8,
        temporal_iou=0.8,
        overlap_score=0.8,
        safety_status="safe",
        safety_reason=None,
        quality_result=TranslationQualityResult(
            accepted=True, status="accepted", reasons=(), cjk_ratio=0.0
        ),
        metadata={},
    )

def test_fit_layout_creates_single_event() -> None:
    layout = _layout(FIT, "fit")
    event = DynamicAssEventService.build_event(layout, _mapping())
    assert event is not None
    assert event.interval_id == "interval_1"
    assert event.start == 1.0
    assert event.end == 2.0
    assert event.text == "Xin chào"
    assert event.font_size == 24
    assert event.position_x == 55.0
    assert event.position_y == 40.0
    assert event.cover_bbox == layout.expanded_cover_bbox
    assert event.replacement_status == "REPLACED"

def test_start_end_preserved() -> None:
    layout = _layout(FIT, "fit")
    event = DynamicAssEventService.build_event(layout, _mapping(start=10.5, end=11.5))
    assert event.start == 10.5
    assert event.end == 11.5

def test_two_lines_ass_line_break() -> None:
    layout = _layout(FIT, "fit", lines=("Xin chào", "Việt Nam"))
    event = DynamicAssEventService.build_event(layout, _mapping())
    assert event.text == "Xin chào\\NViệt Nam"

def test_review_required_returns_none() -> None:
    layout = _layout(REVIEW_REQUIRED, "mapping_not_eligible")
    assert DynamicAssEventService.build_event(layout, _mapping()) is None

def test_skipped_returns_none() -> None:
    layout = _layout(SKIPPED, "true_person_overlap_risk")
    assert DynamicAssEventService.build_event(layout, _mapping()) is None

def test_empty_target_text_returns_none() -> None:
    layout = _layout(FIT, "fit", lines=())
    assert DynamicAssEventService.build_event(layout, _mapping()) is None

def test_build_events_filters_correctly() -> None:
    layouts = [
        _layout(FIT, "fit"),
        _layout(REVIEW_REQUIRED, "mapping_not_eligible"),
        _layout(SKIPPED, "true_person_overlap_risk"),
        _layout(FIT, "fit", lines=("Viet", "Nam")),
    ]
    mappings = [_mapping(), _mapping(), _mapping(), _mapping()]
    events = DynamicAssEventService.build_events(list(zip(layouts, mappings)))
    assert len(events) == 2
    assert events[0].text == "Xin chào"
    assert events[1].text == "Viet\\NNam"
