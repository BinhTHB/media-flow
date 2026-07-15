import pytest
from src.application.services.dynamic_subtitle_text_layout_service import (
    DynamicSubtitleLayoutConfig,
    DynamicSubtitleTextLayoutService,
    FIT,
    REVIEW_REQUIRED,
    SKIPPED,
)
from src.application.services.dynamic_subtitle_timeline_service import (
    TimelineMappingResult,
)
from src.application.services.translation_quality_gate import (
    TranslationQualityResult,
)

class FakeMeasurer:
    """Fake measurer where each character is font_size * 0.5 wide and height is font_size."""
    def measure(
        self,
        text: str,
        *,
        font_family: str,
        font_size: int,
        bold: bool,
        spacing: float,
    ) -> tuple[float, float]:
        w = len(text) * font_size * 0.5
        if bold:
            w *= 1.05
        if len(text) > 1:
            w += spacing * (len(text) - 1)
        return w, float(font_size)

def _mapping(**overrides) -> TimelineMappingResult:
    values = {
        "interval_id": "interval_1",
        "start": 1.0,
        "end": 2.0,
        "source_text": "你好",
        "target_text": "Xin chào",
        "source_bbox": {"left": 10.0, "top": 20.0, "right": 100.0, "bottom": 60.0},
        "expanded_cover_bbox": {"left": 5.0, "top": 15.0, "right": 155.0, "bottom": 85.0},
        "mapping_status": "mapped",
        "mapping_reason": "timestamp_overlap",
        "matched_segment_id": "ocr_1",
        "intersection_duration": 0.8,
        "interval_coverage": 0.8,
        "segment_coverage": 0.8,
        "temporal_iou": 0.8,
        "overlap_score": 0.8,
        "safety_status": "safe",
        "safety_reason": None,
        "quality_result": TranslationQualityResult(
            accepted=True, status="accepted", reasons=(), cjk_ratio=0.0
        ),
        "metadata": {
            "font_size": 24,
            "font_family": "Arial",
            "bold": False,
            "spacing": 0.0,
        },
    }
    values.update(overrides)
    return TimelineMappingResult(**values)

def test_layout_fit_short_latin_one_line() -> None:
    service = DynamicSubtitleTextLayoutService(FakeMeasurer())
    # available width = (155 - 5) - 24 = 126. "Xin chào" is 8 chars, 8 * 24 * 0.5 = 96 <= 126. Fits!
    result = service.layout(_mapping(target_text="Xin chào"), 720, 1280)
    assert result.layout_status == FIT
    assert result.layout_reason == "fit"
    assert result.lines == ("Xin chào",)
    assert result.font_size == 24
    assert result.position_x == 80.0  # (5 + 155)/2
    assert result.position_y == 50.0  # (15 + 85)/2

def test_layout_fit_long_latin_two_lines_balanced() -> None:
    service = DynamicSubtitleTextLayoutService(FakeMeasurer())
    m = _mapping(target_text="Xin chào Việt Nam thân")
    result = service.layout(m, 720, 1280)
    assert result.layout_status == FIT
    assert len(result.lines) == 2
    # At size 16: "Xin chào" (8) and "Việt Nam thân" (13) is chosen as first equal score
    assert result.lines == ("Xin chào", "Việt Nam thân")

def test_layout_reduces_font_to_fit() -> None:
    service = DynamicSubtitleTextLayoutService(FakeMeasurer())
    # box w=100 (max_text_w=76), h=60 (max_text_h=44).
    # forces font size down to 19 to fit 8-char words in 2 lines
    small_box = {"left": 5.0, "top": 15.0, "right": 105.0, "bottom": 75.0}
    m = _mapping(
        target_text="Xin chào Việt Nam",
        expanded_cover_bbox=small_box,
    )
    result = service.layout(m, 720, 1280)
    assert result.layout_status == FIT
    assert result.font_size < 24
    assert result.font_size >= 16

def test_layout_does_not_fit_at_min_font() -> None:
    service = DynamicSubtitleTextLayoutService(FakeMeasurer())
    m = _mapping(
        target_text="Xin chào Việt Nam thân yêu rất nhiều",
        expanded_cover_bbox={"left": 5.0, "top": 15.0, "right": 25.0, "bottom": 35.0},
    )
    result = service.layout(m, 720, 1280)
    assert result.layout_status == REVIEW_REQUIRED
    assert result.layout_reason == "target_text_does_not_fit"

def test_layout_keeps_vietnamese_unicode() -> None:
    service = DynamicSubtitleTextLayoutService(FakeMeasurer())
    result = service.layout(_mapping(target_text="Tiếng Việt"), 720, 1280)
    assert result.layout_status == FIT
    assert "Tiếng" in result.lines[0]

def test_layout_missing_or_zero_area_bbox() -> None:
    service = DynamicSubtitleTextLayoutService(FakeMeasurer())
    res1 = service.layout(_mapping(expanded_cover_bbox=None), 720, 1280)
    assert res1.layout_status == REVIEW_REQUIRED
    assert res1.layout_reason == "missing_cover_bbox"
    zero_box = {"left": 5.0, "top": 15.0, "right": 5.0, "bottom": 15.0}
    res2 = service.layout(_mapping(expanded_cover_bbox=zero_box), 720, 1280)
    assert res2.layout_status == REVIEW_REQUIRED
    assert res2.layout_reason == "missing_cover_bbox"

def test_layout_bbox_out_of_frame() -> None:
    service = DynamicSubtitleTextLayoutService(FakeMeasurer())
    out_box = {"left": 5.0, "top": 15.0, "right": 800.0, "bottom": 65.0}
    res = service.layout(_mapping(expanded_cover_bbox=out_box), 720, 1280)
    assert res.layout_status == REVIEW_REQUIRED
    assert res.layout_reason == "cover_bbox_out_of_frame"

def test_layout_invalid_frame_geometry() -> None:
    service = DynamicSubtitleTextLayoutService(FakeMeasurer())
    res = service.layout(_mapping(), 0, 1280)
    assert res.layout_status == REVIEW_REQUIRED
    assert res.layout_reason == "invalid_frame_geometry"

def test_layout_mapping_not_eligible() -> None:
    service = DynamicSubtitleTextLayoutService(FakeMeasurer())
    res = service.layout(_mapping(mapping_status="review_required"), 720, 1280)
    assert res.layout_status == REVIEW_REQUIRED
    assert res.layout_reason == "mapping_not_eligible"

def test_layout_safety_reason_preserved_skipped() -> None:
    service = DynamicSubtitleTextLayoutService(FakeMeasurer())
    res = service.layout(
        _mapping(safety_status="skipped", safety_reason="true_person_overlap_risk"),
        720, 1280,
    )
    assert res.layout_status == SKIPPED
    assert res.layout_reason == "true_person_overlap_risk"

def test_layout_input_object_not_mutated() -> None:
    service = DynamicSubtitleTextLayoutService(FakeMeasurer())
    m = _mapping()
    original_text = m.target_text
    service.layout(m, 720, 1280)
    assert m.target_text == original_text

def test_layout_picks_largest_font() -> None:
    service = DynamicSubtitleTextLayoutService(FakeMeasurer())
    huge_box = {"left": 5.0, "top": 15.0, "right": 505.0, "bottom": 85.0}
    m = _mapping(target_text="Xin chào", expanded_cover_bbox=huge_box)
    result = service.layout(m, 720, 1280)
    assert result.font_size == 24

def test_layout_position_within_bbox_no_bottom_fallback() -> None:
    service = DynamicSubtitleTextLayoutService(FakeMeasurer())
    m = _mapping(
        expanded_cover_bbox={"left": 100.0, "top": 200.0, "right": 200.0, "bottom": 300.0},
    )
    result = service.layout(m, 720, 1280)
    assert result.layout_status == FIT
    assert result.position_x == 150.0
    assert result.position_y == 250.0

def test_layout_max_lines_respected() -> None:
    config = DynamicSubtitleLayoutConfig(max_lines=1)
    service = DynamicSubtitleTextLayoutService(FakeMeasurer(), config=config)
    m = _mapping(
        target_text="Xin chào Việt Nam",
        expanded_cover_bbox={"left": 5.0, "top": 15.0, "right": 75.0, "bottom": 55.0},
    )
    result = service.layout(m, 720, 1280)
    assert result.layout_status == REVIEW_REQUIRED
    assert result.layout_reason == "target_text_does_not_fit"

def test_layout_punctuation_and_very_long_word() -> None:
    service = DynamicSubtitleTextLayoutService(FakeMeasurer())
    long_word = "A" * 100
    res = service.layout(_mapping(target_text=long_word), 720, 1280)
    assert res.layout_status == REVIEW_REQUIRED
    assert res.layout_reason == "target_text_does_not_fit"
    res2 = service.layout(_mapping(target_text="...!?"), 720, 1280)
    assert res2.layout_status == FIT
