from scripts.render_adaptive_subtitles import (
    _fit_mask_to_text,
    _mask_for_caption,
    _moving_mask_segments,
    _placement_for_caption,
    _coverage_strategy,
    _valid_bbox,
    _target_text,
    _bubble_bbox,
)
from src.application.services.ass_generator import ASSGenerator
from src.domain.entities.segment import Segment
from src.domain.entities.subtitle_style import SubtitleStyle


def _sample(strategy="expanded_line_bbox"):
    return {
        "strategy": strategy,
        "selected_source_bbox": {
            "left": 300, "top": 2000, "right": 1100, "bottom": 2250,
        },
    }


def test_adaptive_mask_is_local_and_centered_in_portrait_frame():
    mask, strategy, reason = _mask_for_caption([_sample()], 1440, 2560)
    placement = _placement_for_caption(mask, strategy)

    assert reason is None
    assert strategy == "expanded_line_bbox"
    assert mask["left"] > 0 and mask["right"] < 1440
    assert placement.x == round((mask["left"] + mask["right"]) / 2)
    assert placement.y == round((mask["top"] + mask["bottom"]) / 2)


def test_missing_detection_uses_explicit_conditional_fallback():
    mask, strategy, reason = _mask_for_caption([], 1440, 2560)

    assert strategy == "full_width_fallback"
    assert reason == "no_coordinate_validated_source_bbox"
    assert mask["right"] == 1440


def test_ass_generates_per_caption_mask_with_portrait_resolution():
    mask, strategy, _ = _mask_for_caption([_sample("local_band")], 1440, 2560)
    placement = _placement_for_caption(mask, strategy)
    segment = Segment(
        id="caption", start=5.0, end=7.0, text="source",
        translated_text="Bản dịch", source="speech", confidence=0.9,
    )

    content = ASSGenerator.generate(
        [segment], SubtitleStyle(alignment=5), [placement], 1440, 2560
    )

    assert "PlayResX: 1440" in content
    assert "PlayResY: 2560" in content
    assert r"\p1\1c&H000000&" in content
    assert rf"\pos({placement.x},{placement.y})" in content


def test_long_target_text_expands_local_mask_without_clipping():
    mask = {"left": 650, "top": 1900, "right": 790, "bottom": 2050}

    fitted, target, clipped = _fit_mask_to_text(
        mask, "Một câu phụ đề dịch dài cần mở rộng vùng cục bộ", 1440, 2560
    )

    assert fitted["left"] <= target["left"] < target["right"] <= fitted["right"]
    assert 0 <= fitted["left"] < fitted["right"] <= 1440
    assert 0 <= fitted["top"] < fitted["bottom"] <= 2560
    assert clipped is False

def test_moving_track_splits_into_local_masks():
    caption = Segment(
        id="moving", start=0.0, end=3.0, text="source",
        translated_text="translated", source="ocr", confidence=1.0,
    )
    samples = [
        {"actual_sampled_timestamp": 0.5, "selected_source_bbox": {"left": 100, "top": 200, "right": 400, "bottom": 240}},
        {"actual_sampled_timestamp": 1.5, "selected_source_bbox": {"left": 100, "top": 400, "right": 400, "bottom": 440}},
        {"actual_sampled_timestamp": 2.5, "selected_source_bbox": {"left": 100, "top": 600, "right": 400, "bottom": 640}},
    ]

    segments = _moving_mask_segments(caption, samples, 540, 960)

    assert len(segments) == 3
    assert segments[0]["mask_bbox"]["bottom"] < segments[-1]["mask_bbox"]["top"]

def test_safe_replace_only_uses_verified_intervals():
    assert _coverage_strategy({
        "decision": "AUTO_REPLACE",
        "layout_mode": "stable_track_subtitle",
        "final_interval_confidence": 0.9,
        "scene_text_risk_score": 0.0,
    }, None, "safe_replace", False) == "floating_caption_source_replace"
    assert _coverage_strategy({"decision": "REVIEW_REQUIRED"}, None, "safe_replace", False) == "skip"
    assert _coverage_strategy(None, None, "safe_replace", False) == "skip"
    assert _coverage_strategy({
        "decision": "AUTO_REPLACE",
        "layout_mode": "stable_track_subtitle",
        "final_interval_confidence": 0.5,
        "scene_text_risk_score": 0.0,
    }, None, "safe_replace", False) == "skip"

def test_full_coverage_fallback_never_claims_source_replacement():
    assert _coverage_strategy({"decision": "SKIP_REPLACEMENT"}, None, "full_coverage_fallback", True) == "fallback_caption_placement"
    assert _coverage_strategy({"decision": "SKIP_REPLACEMENT"}, None, "full_coverage_fallback", False) == "skip"

def test_manual_region_requires_valid_source_bbox():
    manual = {"user_decision": "MANUAL_REGION", "manual_source_bbox": {"left": 1, "top": 2, "right": 10, "bottom": 20}}

    assert _coverage_strategy({"decision": "REVIEW_REQUIRED"}, manual, "full_coverage_review", False) == "manual_region_replace"
    assert _valid_bbox(manual["manual_source_bbox"], 100, 100) is True
    assert _valid_bbox({"left": -1, "top": 2, "right": 10, "bottom": 20}, 100, 100) is False

def test_target_text_never_uses_review_placeholder_and_prefers_asr():
    caption = Segment("review", 0, 1, "[REVIEW]", "[REVIEW]", "ocr", 1.0)

    text, source, mode, missing = _target_text(caption, {"asr_segment_text": "Đúng rồi đĩa này chứ sao?"}, "normalized")

    assert text == "Đúng rồi đĩa này chứ sao?"
    assert source == "asr"
    assert mode == "source_text_echo_for_validation"
    assert missing is False

def test_target_text_falls_back_to_source_ocr_and_reports_missing():
    caption = Segment("review", 0, 1, "[REVIEW]", "[REVIEW]", "ocr", 1.0)

    text, source, _, missing = _target_text(caption, {"selected_candidate": {"raw_ocr_text": "Phụ đề gốc"}}, "asr")
    missing_text, missing_source, _, missing = _target_text(caption, None, "normalized")

    assert (text, source) == ("Phụ đề gốc", "ocr")
    assert (missing_text, missing_source, missing) == ("", None, True)

def test_floating_replace_prefers_ocr_and_expands_bubble_bbox():
    caption = Segment("review", 0, 1, "[REVIEW]", "[REVIEW]", "ocr", 1.0)
    text, source, mode, _ = _target_text(
        caption, {"selected_candidate": {"raw_ocr_text": "Đúng rồi đĩa này chứ sao?"}, "asr_segment_text": "Đúng rồi đĩa này chưa sao?"}, "ocr"
    )
    bubble = _bubble_bbox({"left": 100, "top": 100, "right": 300, "bottom": 150}, 400, 800)

    assert (text, source, mode) == ("Đúng rồi đĩa này chứ sao?", "ocr", "source_caption_ocr_echo_for_validation")
    assert bubble["left"] < 100 and bubble["right"] > 300

def test_source_replacement_mode_never_uses_bottom_fallback():
    assert _coverage_strategy({"decision": "REVIEW_REQUIRED"}, None, "source_replacement_mode", True) == "skip"
    assert _coverage_strategy({"decision": "REVIEW_REQUIRED"}, None, "accessibility_fallback_subtitle_mode", True) == "fallback_caption_placement"
