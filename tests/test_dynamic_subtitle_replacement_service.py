import pytest
from src.application.services.dynamic_subtitle_replacement_service import (
    STATUS_REPLACED,
    STATUS_SKIPPED,
    STATUS_REVIEW,
    STATUS_FALLBACK,
    DynamicSubtitleReplacementService,
)
from src.application.services.dynamic_subtitle_text_layout_service import (
    DynamicSubtitleLayoutResult,
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
from src.application.services.dynamic_ass_event_service import (
    DynamicAssEvent,
)

def _mapping(**overrides) -> TimelineMappingResult:
    values = {
        "interval_id": "interval_1",
        "start": 1.0,
        "end": 2.0,
        "source_text": "你好",
        "target_text": "Xin chào",
        "source_bbox": {"left": 10.0, "top": 20.0, "right": 100.0, "bottom": 60.0},
        "expanded_cover_bbox": {"left": 5.0, "top": 15.0, "right": 105.0, "bottom": 65.0},
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
            "local_safety_override": True,
            "override_reason": "static_region_person_overlap_override",
        },
    }
    values.update(overrides)
    return TimelineMappingResult(**values)

def _layout(status: str = FIT, reason: str = "fit") -> DynamicSubtitleLayoutResult:
    return DynamicSubtitleLayoutResult(
        interval_id="interval_1",
        layout_status=status,
        layout_reason=reason,
        lines=("Xin chào",),
        font_size=24,
        position_x=55.0,
        position_y=40.0,
        text_width=64.0,
        text_height=27.6,
        source_bbox={"left": 10.0, "top": 20.0, "right": 100.0, "bottom": 60.0},
        expanded_cover_bbox={"left": 5.0, "top": 15.0, "right": 105.0, "bottom": 65.0},
    )

def _event() -> DynamicAssEvent:
    return DynamicAssEvent(
        interval_id="interval_1",
        start=1.0,
        end=2.0,
        text="Xin chào",
        lines=("Xin chào",),
        font_size=24,
        position_x=55.0,
        position_y=40.0,
        cover_bbox={"left": 5.0, "top": 15.0, "right": 105.0, "bottom": 65.0},
        replacement_status="REPLACED",
    )

def test_replacement_decision_replaced() -> None:
    service = DynamicSubtitleReplacementService()
    decision = service.decide(_mapping(), _layout(), _event())
    assert decision.status == STATUS_REPLACED
    assert decision.reason == "replacement_ready"
    assert decision.candidate_level_override is True
    assert decision.override_reason == "static_region_person_overlap_override"
    assert decision.ass_event is not None

def test_replacement_decision_skipped_for_hard_safety() -> None:
    service = DynamicSubtitleReplacementService()
    # hard safety mapped
    m = _mapping(safety_status="skipped", safety_reason="true_person_overlap_risk")
    decision = service.decide(m, _layout(SKIPPED, "true_person_overlap_risk"), None)
    assert decision.status == STATUS_SKIPPED
    assert decision.reason == "hard_safety_reject"
    assert decision.safety_reason == "true_person_overlap_risk"
    assert decision.ass_event is None

def test_replacement_decision_review_for_missing_bbox() -> None:
    service = DynamicSubtitleReplacementService()
    m = _mapping(source_bbox=None)
    decision = service.decide(m, _layout(), _event())
    assert decision.status == STATUS_REVIEW
    assert decision.reason == "missing_source_bbox"

def test_replacement_decision_review_for_ambiguous_mapping() -> None:
    service = DynamicSubtitleReplacementService()
    m = _mapping(mapping_status="review_required", mapping_reason="missing_or_ambiguous_timeline_match")
    decision = service.decide(m, _layout(), _event())
    assert decision.status == STATUS_REVIEW
    assert decision.reason == "mapping_not_eligible"

def test_replacement_decision_review_for_quality_fail() -> None:
    service = DynamicSubtitleReplacementService()
    m = _mapping(quality_result=TranslationQualityResult(accepted=False, status="review_required", reasons=("placeholder_target_text",), cjk_ratio=0.0))
    decision = service.decide(m, _layout(), _event())
    assert decision.status == STATUS_REVIEW
    assert decision.reason == "translation_quality_failed"

def test_replacement_decision_review_for_layout_fail() -> None:
    service = DynamicSubtitleReplacementService()
    decision = service.decide(_mapping(), _layout(REVIEW_REQUIRED, "target_text_does_not_fit"), _event())
    assert decision.status == STATUS_REVIEW
    assert decision.reason == "layout_not_fit"

def test_replacement_decision_review_for_missing_event() -> None:
    service = DynamicSubtitleReplacementService()
    decision = service.decide(_mapping(), _layout(), None)
    assert decision.status == STATUS_REVIEW
    assert decision.reason == "missing_ass_event"

def test_replacement_decision_force_static_fallback() -> None:
    service = DynamicSubtitleReplacementService()
    decision = service.decide(_mapping(), _layout(), _event(), force_static_fallback=True)
    assert decision.status == STATUS_FALLBACK
    assert decision.reason == "static_fallback_requested"
    assert decision.ass_event is None

def test_replacement_decision_not_mutating_input() -> None:
    service = DynamicSubtitleReplacementService()
    m = _mapping()
    original_status = m.mapping_status
    service.decide(m, _layout(), _event())
    assert m.mapping_status == original_status