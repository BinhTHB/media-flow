from scripts.render_adaptive_subtitles import (
    _bbox_iou,
    _cover_safety_evidence,
    _cover_target_text,
    _coverage_strategy,
    _padded_cover_bbox,
)
from src.domain.entities.segment import Segment


def _interval(**overrides):
    value = {
        "source_bbox": {"left": 10, "top": 80, "right": 200, "bottom": 120},
        "mask_area_ratio": 0.03,
        "scene_text_risk_score": 0.0,
        "menu_signage_risk_score": 0.0,
        "perspective_attached_text_risk": 0.0,
        "watermark_risk_score": 0.0,
        "person_risk_score": 0.0,
    }
    value.update(overrides)
    return value


def test_cover_mode_uses_placeholder_instead_of_ocr_echo():
    caption = Segment("id", 0, 1, "source", "[REVIEW]", "ocr", 1.0)

    assert _cover_target_text(caption, 1) == (
        "DỊCH_2", "placeholder", "placeholder_target"
    )


def test_cover_mode_uses_translated_target():
    caption = Segment("id", 0, 1, "source", "Bản dịch", "ocr", 1.0)

    assert _cover_target_text(caption, 0) == (
        "Bản dịch", "translated_text", "translated_target"
    )


def test_cover_mode_requires_local_safety_evidence():
    assert _coverage_strategy(
        _interval(), None, "cover_source_subtitle", False,
        frame_width=300, frame_height=200,
    ) == "cover_source_subtitle"
    assert _coverage_strategy(
        _interval(person_risk_score=None), None, "cover_source_subtitle", False,
        frame_width=300, frame_height=200,
    ) == "skip"


def test_cover_mode_requires_localized_source_bbox_and_keeps_high_iou():
    interval = _interval()
    source = interval["source_bbox"]
    cover = _padded_cover_bbox(source, 1440, 2560)

    assert _bbox_iou(source, cover) > 0.15
    assert _coverage_strategy(
        _interval(source_bbox=None), None, "cover_source_subtitle", False,
        frame_width=300, frame_height=200,
    ) == "skip"

def test_cover_mode_overrides_small_hog_person_overlap_only():
    interval = _interval(
        person_risk_score=1.0,
        person_risk_evidence={
            "detected_person_boxes": [
                {"left": 10, "top": 80, "right": 50, "bottom": 100}
            ]
        },
    )

    evidence = _cover_safety_evidence(
        interval, has_target_text=True, width=300, height=200
    )
    assert evidence["local_safety_override"] is True
    assert evidence["override_reason"] == "hog_person_overlap_only_false_reject"

def test_cover_mode_keeps_large_person_and_scene_risks_rejected():
    interval = _interval(
        person_risk_score=1.0,
        person_risk_evidence={
            "detected_person_boxes": [
                {"left": 10, "top": 80, "right": 200, "bottom": 120}
            ]
        },
    )
    assert _cover_safety_evidence(
        interval, has_target_text=True, width=300, height=200
    )["keep_reject_reason"] == "true_person_overlap_risk"
    assert _cover_safety_evidence(
        _interval(scene_text_risk_score=0.20),
        has_target_text=True, width=300, height=200,
    )["keep_reject_reason"] == "non_person_overlap_risk"

def test_static_region_override_allows_non_total_person_overlap():
    evidence = _cover_safety_evidence(
        _interval(
            stability_score=0.90,
            static_region_found=True,
            fixed_bottom_fallback=False,
            person_risk_score=1.0,
            person_risk_evidence={
                "detected_person_boxes": [
                    {"left": 10, "top": 80, "right": 150, "bottom": 120}
                ]
            },
        ),
        has_target_text=True, width=300, height=200,
    )

    assert evidence["allowed"] is True
    assert evidence["static_region_person_overlap_override"] is True
    assert evidence["override_reason"] == "static_region_person_overlap_override"

def test_static_region_override_rejects_total_person_overlap():
    evidence = _cover_safety_evidence(
        _interval(
            stability_score=0.90,
            static_region_found=True,
            fixed_bottom_fallback=False,
            person_risk_score=1.0,
            person_risk_evidence={
                "detected_person_boxes": [
                    {"left": 10, "top": 80, "right": 200, "bottom": 120}
                ]
            },
        ),
        has_target_text=True, width=300, height=200,
    )

    assert evidence["allowed"] is False
    assert evidence["keep_reject_reason"] == "true_person_overlap_risk"

