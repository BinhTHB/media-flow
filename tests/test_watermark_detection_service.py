from src.application.services.adaptive_subtitle_region_service import BoundingBox
from src.application.services.watermark_detection_service import (
    WatermarkDetectionService,
    WatermarkObservation,
)


def _observations(
    count: int,
    bbox: BoundingBox = BoundingBox(10, 10, 150, 70),
    **values,
) -> list[WatermarkObservation]:
    return [
        WatermarkObservation(float(index), bbox, **values)
        for index in range(count)
    ]


def _detect(observations, sampled=10):
    return WatermarkDetectionService().detect(observations, sampled, 1000, 600)


def test_persistent_corner_logo_is_classified_as_watermark():
    candidate = _detect(_observations(10, overlay_style_score=0.9))[0]

    assert candidate["classification"] == "logo"
    assert candidate["watermark_score"] >= 0.67


def test_persistent_corner_text_is_classified_as_watermark():
    candidate = _detect(_observations(
        8, text_content="CHANNEL", ocr_confidence=0.95, overlay_style_score=0.9
    ))[0]

    assert candidate["classification"] == "watermark"
    assert candidate["text_content"] == "CHANNEL"


def test_changing_subtitle_text_is_not_watermark():
    observations = [
        WatermarkObservation(
            float(index), BoundingBox(200, 500, 800, 570),
            text_content=f"dialogue {index}", asr_match_score=0.9,
            subtitle_overlap_score=1.0, overlay_style_score=0.8,
        )
        for index in range(10)
    ]

    assert all(item["classification"] == "rejected" for item in _detect(observations))


def test_floating_caption_is_not_watermark():
    observations = [
        WatermarkObservation(
            float(index), BoundingBox(100 + index * 60, 150, 360 + index * 60, 210),
            text_content="caption", asr_match_score=0.8, subtitle_overlap_score=0.9,
        )
        for index in range(8)
    ]

    assert all(item["classification"] == "rejected" for item in _detect(observations))


def test_wall_or_menu_text_is_uncertain_and_not_automatically_blurred():
    candidate = _detect(_observations(
        10, BoundingBox(250, 120, 750, 420),
        text_content="MENU", scene_text_risk_score=0.9,
    ))[0]

    assert candidate["classification"] == "uncertain"


def test_groups_stable_watermark_bbox_across_frames():
    observations = [
        WatermarkObservation(
            float(index), BoundingBox(10 + index % 2, 10, 150 + index % 2, 70),
            overlay_style_score=0.9,
        )
        for index in range(10)
    ]

    candidates = _detect(observations)

    assert len(candidates) == 1
    assert candidates[0]["frame_occurrences"] == 10
    assert candidates[0]["position_stability_score"] > 0.99


def test_low_persistence_candidate_is_rejected():
    candidate = _detect(_observations(3), sampled=10)[0]

    assert candidate["classification"] == "rejected"
    assert "low_frame_coverage" in candidate["rejected_reasons"]


def test_blur_bbox_is_clamped_inside_frame_bounds():
    box = WatermarkDetectionService.clamp_bbox(
        BoundingBox(-10, -5, 1010, 605), 1000, 600
    )

    assert box == BoundingBox(0, 0, 1000, 600)


def test_raw_fragments_are_merged_before_final_visualization():
    candidates = _detect(
        _observations(10, BoundingBox(10, 10, 80, 50), overlay_style_score=0.9)
        + _observations(10, BoundingBox(85, 10, 150, 50), overlay_style_score=0.9)
    )

    final = WatermarkDetectionService.merge_nearby(candidates, 1000, 600)

    assert len(final) == 1
    assert len(final[0]["merged_candidate_ids"]) == 2


def test_table_texture_candidate_is_rejected():
    observations = [
        WatermarkObservation(
            float(index), BoundingBox(200 + index * 25, 180, 500 + index * 25, 400),
            scene_text_risk_score=0.8,
        )
        for index in range(10)
    ]

    assert all(item["classification"] != "logo" for item in _detect(observations))


def test_subtitle_bubble_candidate_is_rejected():
    candidate = _detect(_observations(
        10, BoundingBox(100, 480, 900, 590),
        subtitle_overlap_score=1.0, asr_match_score=0.9,
    ))[0]

    assert candidate["classification"] == "rejected"


def test_final_candidate_count_is_bounded_by_grouping():
    candidates = _detect(
        _observations(10, BoundingBox(10, 10, 80, 50), overlay_style_score=0.9)
        + _observations(10, BoundingBox(85, 10, 150, 50), overlay_style_score=0.9)
        + _observations(2, BoundingBox(600, 200, 700, 250))
    )

    final = WatermarkDetectionService.merge_nearby(
        [item for item in candidates if item["classification"] != "rejected"],
        1000, 600,
    )

    assert len(final) <= 2

def _safety_candidate(**overrides):
    candidate = {
        "candidate_id": "candidate",
        "bbox": {"left": 10, "top": 10, "right": 150, "bottom": 70},
        "position_stability_score": 0.99,
        "size_stability_score": 0.99,
        "corner_or_edge_score": 0.95,
        "scene_attachment_risk": 0.05,
        "face_person_overlap_risk": 0.0,
        "classification": "logo",
        "rejected_reasons": [],
        "merged_candidate_ids": ["candidate"],
    }
    candidate.update(overrides)
    return candidate

def test_stable_corner_overlay_passes_balanced_safety():
    candidate = WatermarkDetectionService.apply_auto_blur_safety(
        _safety_candidate(), 1000, 600, 1, "balanced"
    )

    assert candidate["balanced_auto_blur_allowed"] is True
    assert candidate["overlay_independence_score"] > 0.95

def test_person_region_candidate_is_rejected_in_balanced_profile():
    candidate = WatermarkDetectionService.apply_auto_blur_safety(
        _safety_candidate(face_person_overlap_risk=0.8), 1000, 600, 1, "balanced"
    )

    assert candidate["balanced_auto_blur_allowed"] is False
    assert candidate["rejection_reason"] == "person_overlap_risk"

def test_clothing_or_object_attached_logo_is_not_auto_blurred():
    candidate = WatermarkDetectionService.apply_auto_blur_safety(
        _safety_candidate(size_stability_score=0.55), 1000, 600, 1, "balanced"
    )

    assert candidate["classification"] == "uncertain"
    assert candidate["object_attachment_risk"] > 0.30

def test_moving_candidate_fails_overlay_independence():
    candidate = WatermarkDetectionService.apply_auto_blur_safety(
        _safety_candidate(position_stability_score=0.75), 1000, 600, 1, "balanced"
    )

    assert candidate["balanced_auto_blur_allowed"] is False
    assert candidate["screen_position_stability_score"] == 0.75

def test_multiple_final_candidates_trigger_balanced_review():
    candidate = WatermarkDetectionService.apply_auto_blur_safety(
        _safety_candidate(), 1000, 600, 2, "balanced"
    )

    assert candidate["balanced_auto_blur_allowed"] is False
    assert candidate["review_reason"] == "multiple_distinct_candidates"
