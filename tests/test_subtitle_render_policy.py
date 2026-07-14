from src.application.services.subtitle_render_policy import SubtitleRenderPolicy
from scripts.render_subtitle_policy import _static_style, _write_sidecars
from src.domain.entities.segment import Segment


def _interval(**overrides):
    value = {
        "decision": "AUTO_REPLACE",
        "layout_mode": "stable_track_subtitle",
        "final_interval_confidence": 0.9,
        "scene_text_risk_score": 0.0,
        "source_subtitle_present_score": 1.0,
        "target_text_fit_score": 1.0,
    }
    value.update(overrides)
    return value


def test_confident_stable_interval_allows_source_replacement():
    decision = SubtitleRenderPolicy().decide(
        _interval(), has_target_text=True, resource_profile="balanced"
    )

    assert decision.decision == "AUTO_SOURCE_REPLACE"

def test_balanced_profile_keeps_static_render_independent():
    policy = SubtitleRenderPolicy()
    replacement = policy.decide(
        _interval(decision="REVIEW_REQUIRED", final_interval_confidence=0.4),
        has_target_text=True,
        resource_profile="balanced",
    )
    static = policy.static_render_gate(
        target_timeline_exists=True,
        timing_valid=True,
        output_mode_includes_static=True,
    )

    assert replacement.decision != "AUTO_SOURCE_REPLACE"
    assert static.decision == "STATIC_SUBTITLE_RENDER"
def test_balanced_dynamic_or_uncertain_interval_requires_review():
    decision = SubtitleRenderPolicy().decide(
        _interval(decision="REVIEW_REQUIRED", final_interval_confidence=0.7),
        has_target_text=True,
        resource_profile="balanced",
    )

    assert decision.decision == "REVIEW_REQUIRED"


def test_skip_replacement_uses_static_fallback_with_target_text():
    decision = SubtitleRenderPolicy().decide(
        _interval(decision="SKIP_REPLACEMENT"),
        has_target_text=True,
        resource_profile="balanced",
    )

    assert decision.decision == "STATIC_FALLBACK"


def test_missing_target_text_skips_all_rendering():
    decision = SubtitleRenderPolicy().decide(
        _interval(), has_target_text=False, resource_profile="balanced"
    )

    assert decision.decision == "SKIP_REPLACEMENT"


def test_low_profile_requires_higher_source_replace_confidence():
    decision = SubtitleRenderPolicy().decide(
        _interval(final_interval_confidence=0.85),
        has_target_text=True,
        resource_profile="low",
    )

    assert decision.decision == "STATIC_FALLBACK"


def test_confident_bottom_lane_is_replacement_candidate():
    decision = SubtitleRenderPolicy().ux_decide(
        _interval(source_bbox={"left": 10, "top": 500, "right": 900, "bottom": 580}),
        "balanced",
    )

    assert decision.decision == "bottom_subtitle_replace_candidate"


def test_visible_original_suppresses_static_fallback_burn_in():
    decision = SubtitleRenderPolicy().ux_decide(
        _interval(decision="REVIEW_REQUIRED", source_bbox=None),
        "balanced",
    )

    assert decision.decision == "static_fallback_suppressed_due_to_duplicate_risk"


def test_missing_source_region_allows_sidecar_fallback():
    decision = SubtitleRenderPolicy().ux_decide(None, "balanced")

    assert decision.decision == "static_fallback_allowed"


def test_karaoke_word_requires_review():
    decision = SubtitleRenderPolicy().ux_decide(
        _interval(layout_mode="karaoke_subtitle"), "balanced"
    )

    assert decision.decision == "karaoke_word_keep_or_review"

def test_static_render_does_not_require_source_replacement():
    policy = SubtitleRenderPolicy()
    replacement = policy.decide(
        _interval(decision="REVIEW_REQUIRED", final_interval_confidence=0.1),
        has_target_text=True,
        resource_profile="balanced",
    )
    static = policy.static_render_gate(
        target_timeline_exists=True,
        timing_valid=True,
        output_mode_includes_static=True,
    )

    assert replacement.decision != "AUTO_SOURCE_REPLACE"
    assert static.decision == "STATIC_SUBTITLE_RENDER"

def test_duplicate_risk_does_not_block_explicit_static_render():
    policy = SubtitleRenderPolicy()
    duplicate = policy.ux_decide(
        _interval(decision="REVIEW_REQUIRED", source_bbox=None), "balanced"
    )
    static = policy.static_render_gate(
        target_timeline_exists=True,
        timing_valid=True,
        output_mode_includes_static=True,
    )

    assert duplicate.decision == "static_fallback_suppressed_due_to_duplicate_risk"
    assert static.decision == "STATIC_SUBTITLE_RENDER"

def test_static_render_gate_reports_supported_block_reasons():
    policy = SubtitleRenderPolicy()

    missing = policy.static_render_gate(
        target_timeline_exists=False,
        timing_valid=True,
        output_mode_includes_static=True,
    )
    invalid = policy.static_render_gate(
        target_timeline_exists=True,
        timing_valid=False,
        output_mode_includes_static=True,
    )
    excluded = policy.static_render_gate(
        target_timeline_exists=True,
        timing_valid=True,
        output_mode_includes_static=False,
    )

    assert missing.reason == "missing_target_timeline"
    assert invalid.reason == "invalid_timing"
    assert excluded.reason == "output_mode_does_not_include_static"
    assert all(
        decision.reason != "source_replace_confidence_low"
        for decision in (missing, invalid, excluded)
    )

def test_replacement_gate_fails_closed_without_confident_intervals():
    policy = SubtitleRenderPolicy()
    blocked = policy.replacement_gate(
        confident_interval_count=0,
        output_mode_includes_replacement=True,
    )
    allowed = policy.replacement_gate(
        confident_interval_count=1,
        output_mode_includes_replacement=True,
    )
    assert (blocked.decision, blocked.reason) == (
        "BLOCK_SOURCE_REPLACE",
        "no_high_confidence_source_replace_intervals",
    )
    assert allowed.decision == "AUTO_SOURCE_REPLACE"

def test_sidecar_gate_is_independent_from_replacement_gate():
    policy = SubtitleRenderPolicy()
    replacement = policy.replacement_gate(
        confident_interval_count=0,
        output_mode_includes_replacement=True,
    )
    sidecars = policy.sidecar_generation_gate(target_timeline_exists=True)

    assert replacement.decision == "BLOCK_SOURCE_REPLACE"
    assert sidecars.decision == "GENERATE_SIDECARS"

def test_sidecar_generation_writes_ass_and_srt(tmp_path):
    style, placement = _static_style("outline", "bottom")
    paths = _write_sidecars(
        tmp_path,
        [Segment("one", 0.0, 1.0, "Source", "Target", "speech", 1.0)],
        style,
        [placement],
    )

    assert set(paths) == {"srt", "ass"}
    assert all(__import__("pathlib").Path(path).is_file() for path in paths.values())
    assert paths["srt"].endswith("subtitles\\output.srt")
    assert paths["ass"].endswith("subtitles\\output.ass")

def test_adaptive_source_replace_accepts_v5_style_floating_candidate():
    decision = SubtitleRenderPolicy().adaptive_decide(
        _interval(
            layout_mode="floating_caption_subtitle",
            final_interval_confidence=0.72,
            mask_area_ratio=0.03,
            selected_candidate={"bbox": {"left": 1, "top": 2, "right": 3, "bottom": 4}},
        ),
        has_target_text=True,
    )

    assert decision.decision == "AUTO_SOURCE_REPLACE"

def test_adaptive_source_replace_rejects_explicit_safety_risks():
    policy = SubtitleRenderPolicy()
    base = {
        "layout_mode": "floating_caption_subtitle",
        "final_interval_confidence": 0.72,
        "mask_area_ratio": 0.03,
        "selected_candidate": {"bbox": {}},
    }

    assert policy.adaptive_decide(
        _interval(**base, person_risk_score=0.2), has_target_text=True
    ).reason == "person_risk"
    assert policy.adaptive_decide(
        _interval(**base, watermark_risk_score=0.2), has_target_text=True
    ).reason == "watermark_risk"
    assert policy.adaptive_decide(
        _interval(**base, scene_text_risk_score=0.2), has_target_text=True
    ).reason == "scene_text_risk"
    assert policy.adaptive_decide(
        _interval(**(base | {"mask_area_ratio": 0.21})), has_target_text=True
    ).reason == "mask_area_above_0_20"

def _safe_override_interval(**overrides):
    value = _interval(
        decision="REVIEW_REQUIRED",
        reason="insufficient_track_evidence",
        review_reason="insufficient_track_evidence",
        mask_area_ratio=0.03,
        menu_signage_risk_score=0.0,
        perspective_attached_text_risk=0.0,
        watermark_risk_score=0.0,
        person_risk_score=0.0,
        selected_candidate={
            "bbox": {"left": 10, "top": 20, "right": 100, "bottom": 60},
            "final_candidate_score": 0.96,
        },
    )
    value.update(overrides)
    return value

def test_adaptive_candidate_override_accepts_safe_local_evidence():
    decision = SubtitleRenderPolicy().adaptive_decide(
        _safe_override_interval(), has_target_text=True
    )

    assert (decision.decision, decision.reason) == (
        "AUTO_SOURCE_REPLACE",
        "candidate_level_override",
    )

def test_adaptive_candidate_override_is_limited_to_allowed_rejections():
    decision = SubtitleRenderPolicy().adaptive_decide(
        _safe_override_interval(
            reason="coordinate_mapping_failed",
            review_reason="coordinate_mapping_failed",
        ),
        has_target_text=True,
    )

    assert decision.reason == "original_rejection_not_override_eligible"

def test_adaptive_candidate_override_fails_closed_without_person_evidence():
    interval = _safe_override_interval()
    interval.pop("person_risk_score")
    decision = SubtitleRenderPolicy().adaptive_decide(
        interval, has_target_text=True
    )

    assert decision.reason == "missing_person_risk_evidence"

def test_adaptive_candidate_override_checks_all_local_safety_risks():
    policy = SubtitleRenderPolicy()

    assert policy.adaptive_decide(
        _safe_override_interval(menu_signage_risk_score=0.2),
        has_target_text=True,
    ).reason == "menu_signage_risk"
    assert policy.adaptive_decide(
        _safe_override_interval(perspective_attached_text_risk=0.2),
        has_target_text=True,
    ).reason == "wall_sign_risk"
    assert policy.adaptive_decide(
        _safe_override_interval(person_risk_score=0.2),
        has_target_text=True,
    ).reason == "person_risk"
