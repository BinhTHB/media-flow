from src.application.services.dynamic_subtitle_timeline_service import (
    DynamicSourceInterval,
    DynamicSubtitleTimelineService,
    TranslatedSourceSegment,
)


BOX = {"left": 10.0, "top": 20.0, "right": 100.0, "bottom": 60.0}
COVER = {"left": 5.0, "top": 15.0, "right": 105.0, "bottom": 65.0}


def _interval(**overrides) -> DynamicSourceInterval:
    values = {
        "interval_id": "interval_1",
        "start": 1.0,
        "end": 2.0,
        "source_text": "你好",
        "source_bbox": BOX,
        "expanded_cover_bbox": COVER,
        "safety_status": "safe",
        "safety_reason": None,
    }
    values.update(overrides)
    return DynamicSourceInterval(**values)


def _segment(**overrides) -> TranslatedSourceSegment:
    values = {
        "segment_id": "ocr_1",
        "start": 1.1,
        "end": 1.9,
        "source_text": "你好",
        "target_text": "Xin chào",
        "provenance": "ocr_source_text",
    }
    values.update(overrides)
    return TranslatedSourceSegment(**values)


def test_maps_unique_ocr_translation_and_preserves_geometry() -> None:
    result = DynamicSubtitleTimelineService().map(
        [_interval()], [_segment()]
    )[0]

    assert result.mapping_status == "mapped"
    assert result.mapping_reason == "timestamp_overlap"
    assert result.matched_segment_id == "ocr_1"
    assert result.target_text == "Xin chào"
    assert result.source_bbox == BOX
    assert result.expanded_cover_bbox == COVER


def test_selects_clearly_better_overlap_candidate() -> None:
    result = DynamicSubtitleTimelineService().map(
        [_interval()],
        [_segment(segment_id="weak", start=0.9, end=1.2), _segment()],
    )[0]

    assert result.mapping_status == "mapped"
    assert result.matched_segment_id == "ocr_1"


def test_marks_near_equal_overlap_as_review_required() -> None:
    result = DynamicSubtitleTimelineService(ambiguity_margin=0.10).map(
        [_interval()],
        [
            _segment(segment_id="left", start=0.9, end=1.5),
            _segment(segment_id="right", start=1.5, end=2.1),
        ],
    )[0]

    assert result.mapping_status == "review_required"
    assert result.mapping_reason == "missing_or_ambiguous_timeline_match"


def test_marks_no_overlap_as_review_required() -> None:
    result = DynamicSubtitleTimelineService().map(
        [_interval()], [_segment(start=3.0, end=4.0)]
    )[0]

    assert result.mapping_status == "review_required"
    assert result.mapping_reason == "missing_or_ambiguous_timeline_match"


def test_rejects_missing_bbox_or_source_text() -> None:
    missing_bbox = DynamicSubtitleTimelineService().map(
        [_interval(source_bbox=None)], [_segment()]
    )[0]
    missing_text = DynamicSubtitleTimelineService().map(
        [_interval(source_text="")], [_segment()]
    )[0]

    assert missing_bbox.mapping_reason == "missing_source_bbox"
    assert missing_text.mapping_reason == "missing_source_text"


def test_rejects_invalid_translation_and_asr_fallback() -> None:
    invalid_text = DynamicSubtitleTimelineService().map(
        [_interval()], [_segment(target_text="DỊCH_1")]
    )[0]
    asr_fallback = DynamicSubtitleTimelineService().map(
        [_interval()], [_segment(provenance="asr_fallback")]
    )[0]

    assert invalid_text.mapping_reason == "translation_quality_rejected"
    assert asr_fallback.mapping_reason == "unsupported_source_provenance"


def test_output_is_deterministic_and_does_not_lose_intervals() -> None:
    results = DynamicSubtitleTimelineService().map(
        [_interval(interval_id="late", start=3.0, end=4.0), _interval(interval_id="early", start=1.0, end=2.0)],
        [_segment(segment_id="late", start=3.1, end=3.9), _segment(segment_id="early", start=1.1, end=1.9)],
    )

    assert [result.interval_id for result in results] == ["early", "late"]
    assert [result.target_text for result in results] == ["Xin chào", "Xin chào"]


def test_preserves_sample5_hard_safety_rejection() -> None:
    result = DynamicSubtitleTimelineService().map(
        [_interval(interval_id="interval_6", safety_status="skipped", safety_reason="true_person_overlap_risk")],
        [_segment()],
    )[0]

    assert result.mapping_status == "review_required"
    assert result.mapping_reason == "true_person_overlap_risk"
    assert result.safety_reason == "true_person_overlap_risk"
