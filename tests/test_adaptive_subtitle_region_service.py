from src.application.services.adaptive_subtitle_region_service import (
    AdaptiveSubtitleRegionService,
    BoundingBox,
    TextBox,
)
from src.infrastructure.artifacts.detailed_ocr_box_cache import (
    DetailedOCRBoxCacheReader,
    DetailedOCRBoxCacheWriter,
)


def _box(left, top, right, bottom, confidence=0.9, text="word"):
    return TextBox(BoundingBox(left, top, right, bottom), confidence, text)


def test_groups_tokens_into_one_line_by_adaptive_y_center():
    lines = AdaptiveSubtitleRegionService().group_lines([
        _box(100, 400, 220, 440, text="one"),
        _box(230, 406, 390, 446, text="line"),
    ])

    assert len(lines) == 1
    assert lines[0].to_dict()["line_text"] == "one line"


def test_splits_and_groups_two_subtitle_lines_into_one_block():
    service = AdaptiveSubtitleRegionService()
    lines = service.group_lines([
        _box(100, 400, 300, 440),
        _box(100, 460, 320, 500),
    ])

    blocks = service.group_blocks(lines)

    assert len(lines) == 2
    assert len(blocks) == 1
    assert blocks[0].bbox.to_dict() == {"left": 100, "top": 400, "right": 320, "bottom": 500}


def test_ignores_tiny_high_confidence_and_keeps_wide_subtitle_line():
    service = AdaptiveSubtitleRegionService()
    selected, ignored = service.filter_boxes(
        [_box(10, 20, 18, 25, 0.99), _box(100, 400, 900, 450)],
        1000, 600,
    )

    assert selected[0].bbox.width == 800
    assert ignored[0]["ignored_reason"] == "tiny_isolated_box"


def test_aggregate_rejects_temporal_outlier():
    bbox, consistency = AdaptiveSubtitleRegionService().aggregate([
        BoundingBox(100, 400, 800, 460),
        BoundingBox(105, 402, 805, 462),
        BoundingBox(10, 10, 30, 30),
    ])

    assert bbox == BoundingBox(102.5, 401, 802.5, 461)
    assert consistency == 2 / 3


def test_selects_expanded_local_and_conditional_fallback_strategies():
    service = AdaptiveSubtitleRegionService()
    bbox = BoundingBox(100, 400, 800, 460)

    assert service.select_strategy(bbox, 0.9, 1000, 600) == "expanded_line_bbox"
    assert service.select_strategy(bbox, 0.6, 1000, 600) == "local_band"
    assert service.select_strategy(None, 0.0, 1000, 600) == "full_width_fallback"


def test_full_width_fallback_never_passes_production_style_gate():
    service = AdaptiveSubtitleRegionService()

    assert service.production_style_replacement("full_width_fallback", 0.10) == "CONDITIONAL"
    assert service.production_style_replacement("expanded_line_bbox", 0.08) == "PASS"
    assert service.production_style_replacement("local_band", 0.18) == "PASS"
    assert service.production_style_replacement("local_band", 0.21) == "CONDITIONAL"


def test_reliability_rewards_stable_wide_subtitle_block():
    service = AdaptiveSubtitleRegionService()
    block = service.group_blocks(service.group_lines([
        _box(100, 400, 300, 440),
        _box(320, 400, 800, 440),
    ]))[0]

    assert service.reliability_score(block, 1.0, 1000, 600, True) >= 0.8


def test_detailed_ocr_cache_round_trip(tmp_path):
    path = tmp_path / "debug" / "precise_subtitle_localization.json"
    records = [{
        "frame_timestamp": 5.0,
        "frame_index": 150,
        "frame_width": 1440,
        "frame_height": 2560,
        "source_image_path": "frame.jpg",
        "provider": "paddleocr",
        "provider_version": "3",
        "device": "gpu",
        "coordinate_space": "source_frame_pixels",
        "boxes": [_box(100, 1200, 900, 1260).to_dict()],
    }]

    DetailedOCRBoxCacheWriter().write(str(path), records)

    assert DetailedOCRBoxCacheReader().read(str(path)) == records


def test_candidate_classifier_rejects_punctuation_and_decorative_symbols():
    service = AdaptiveSubtitleRegionService()

    assert service.classify_candidate("?")["ignored_reason"] == "punctuation_only"
    assert service.classify_candidate("...")["ignored_reason"] == "punctuation_only"
    assert service.classify_candidate("A")["ignored_reason"] == "too_short_isolated_text"
    assert service.classify_candidate("subtitle")["text_classification"] == "meaningful_text"

def test_classifies_stationary_and_smooth_moving_tracks():
    service = AdaptiveSubtitleRegionService()
    stationary = service.classify_track([
        (0.0, BoundingBox(100, 400, 500, 440)),
        (1.0, BoundingBox(100, 402, 500, 442)),
        (2.0, BoundingBox(100, 401, 500, 441)),
    ], 960)
    moving = service.classify_track([
        (0.0, BoundingBox(100, 200, 500, 240)),
        (1.0, BoundingBox(100, 400, 500, 440)),
        (2.0, BoundingBox(100, 600, 500, 640)),
    ], 960)

    assert stationary["track_mode"] == "stationary_track"
    assert moving["track_mode"] == "moving_track"
    assert moving["moving_track_score"] == 1.0

def test_interpolates_missing_moving_track_bbox_without_median_reset():
    service = AdaptiveSubtitleRegionService()
    samples = [
        (0.0, BoundingBox(100, 200, 500, 240)),
        (2.0, BoundingBox(100, 600, 500, 640)),
    ]
    track = service.classify_track(samples, 960)

    result = service.interpolate_track_bbox(1.0, samples, track)

    assert result == BoundingBox(100, 400, 500, 440)

def test_one_frame_jump_is_not_classified_as_moving_track():
    track = AdaptiveSubtitleRegionService().classify_track([
        (0.0, BoundingBox(100, 400, 500, 440)),
        (1.0, BoundingBox(100, 100, 500, 140)),
        (2.0, BoundingBox(100, 400, 500, 440)),
    ], 960)

    assert track["track_mode"] == "unreliable_track"

def test_large_central_candidate_is_not_production_style_safe():
    service = AdaptiveSubtitleRegionService()
    features = service.candidate_features(
        "MENU", BoundingBox(100, 100, 900, 550), 1000, 600, persistence_score=1.0
    )

    assert features["scene_text_risk_score"] == 1.0
    assert features["central_content_overlap_risk"] is True
    assert service.production_style_replacement("local_band", 0.36) == "FAIL"

def test_video_eligibility_allows_stable_track_and_skips_unsupported_video():
    service = AdaptiveSubtitleRegionService()
    stable = service.video_eligibility({
        "total_intervals": 10, "reliable_subtitle_intervals": 9,
        "no_subtitle_skip_intervals": 1, "scene_text_rejection_count": 0,
        "track_jump_count": 0, "max_mask_area_ratio": 0.08,
    })
    unsupported = service.video_eligibility({
        "total_intervals": 20, "reliable_subtitle_intervals": 7,
        "no_subtitle_skip_intervals": 11, "scene_text_rejection_count": 8,
        "track_jump_count": 12, "max_mask_area_ratio": 0.28,
    })

    assert stable["eligibility_decision"] == "AUTO_REPLACE"
    assert unsupported["eligibility_decision"] == "SKIP_REPLACEMENT"

def test_asr_consistency_is_optional_and_does_not_force_visual_selection():
    service = AdaptiveSubtitleRegionService()

    assert service.asr_ocr_consistency("subtitle", None)["asr_available"] is False
    assert service.asr_ocr_consistency("subtitle", "subtitle", 0.9)["asr_ocr_consistency_score"] > 0.9

def test_floating_caption_position_jump_can_auto_replace():
    result = AdaptiveSubtitleRegionService.interval_eligibility({
        "layout_mode": "floating_caption_subtitle", "source_subtitle_present": True,
        "mask_area_ratio": 0.05, "coordinate_mapping": "PASS", "candidate_duration_score": 1.0,
        "asr_text": "hello caption", "asr_confidence": 0.9, "rejected_as_jump": True,
        "candidates": [{"selected": True, "raw_ocr_text": "hello caption", "overlay_style_score": 1.0,
                        "text_box_style_score": 1.0, "caption_box_background_score": 1.0}],
    })

    assert result["decision"] == "AUTO_REPLACE"
    assert result["selected_as_floating_caption"] is True

def test_persistent_scene_text_is_rejected_despite_ocr_confidence():
    features = AdaptiveSubtitleRegionService.floating_caption_features(
        {"persistence_score": 1.0, "overlay_style_score": 0.2}, 0.0, 1.0
    )

    assert features["rejected_as_scene_text"] is True
    assert features["rejection_reason"] == "persistent_scene_text"

def test_menu_signage_and_watermark_are_rejected():
    service = AdaptiveSubtitleRegionService()

    assert service.floating_caption_features({"menu_signage_risk_score": 1.0}, 0.0)["rejection_reason"] == "wall_menu_signage"
    assert service.floating_caption_features({"watermark_risk_score": 1.0}, 0.0)["rejection_reason"] == "watermark_or_logo"

def test_asr_match_boosts_floating_overlay_caption():
    service = AdaptiveSubtitleRegionService()
    weak = service.floating_caption_features({"overlay_style_score": 0.8}, 0.0, 0.8)
    matched = service.floating_caption_features({"overlay_style_score": 0.8}, 1.0, 0.8)

    assert matched["floating_caption_score"] > weak["floating_caption_score"]

def test_position_jump_without_floating_evidence_stays_review_required():
    result = AdaptiveSubtitleRegionService.interval_eligibility({
        "layout_mode": "floating_caption_subtitle", "source_subtitle_present": True,
        "mask_area_ratio": 0.05, "rejected_as_jump": True,
        "candidates": [{"selected": True, "raw_ocr_text": "unknown"}],
    })

    assert result["decision"] == "REVIEW_REQUIRED"

def test_layout_classifier_detects_multiple_caption_positions():
    result = AdaptiveSubtitleRegionService.classify_layout_mode([
        {"selected_source_bbox": {"top": 50, "bottom": 100}, "frame_height": 1000},
        {"selected_source_bbox": {"top": 750, "bottom": 800}, "frame_height": 1000},
    ])

    assert result["layout_mode"] == "floating_caption_subtitle"
    assert result["position_transition_count"] == 1

def test_no_reliable_floating_candidate_skips_without_guessed_region():
    result = AdaptiveSubtitleRegionService.interval_eligibility({
        "layout_mode": "floating_caption_subtitle", "source_subtitle_present": False,
        "strategy": "no_subtitle_skip", "candidates": [],
    })

    assert result["decision"] == "SKIP_REPLACEMENT"
