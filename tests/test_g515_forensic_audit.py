import pytest

from scripts.run_g514_sample2 import validate_visual_qa_result
from scripts.run_g515_forensic_audit import (
    G514ValidationError,
    active_dialogues,
    correct_timing_boundaries,
    parse_container_policy,
    validate_dynamic_ass,
    validate_visual_result,
)


def _dialogue(event_id, layer, role, start=0.0, end=1.0):
    return {
        "event_id": event_id,
        "layer": layer,
        "role": role,
        "start": start,
        "end": end,
        "text": "replacement",
    }


def test_banner_visibility_is_not_a_hard_fail():
    result = validate_visual_qa_result({
        "status": "PASS", "source_text_visible": False, "banner_visible": True,
        "has_duplicate": False, "has_overflow": False, "has_damage": False,
    })
    assert result["status"] == "PASS"


def test_source_glyph_leakage_still_fails():
    result = validate_visual_qa_result({
        "status": "PASS", "source_text_visible": True, "banner_visible": False,
        "has_duplicate": False, "has_overflow": False, "has_damage": False,
    })
    assert result["status"] == "FAIL"


def test_two_active_replacements_in_same_lane_are_detected():
    active = active_dialogues([
        _dialogue("e0", 1, "replacement_text", 0.0, 2.0),
        _dialogue("e1", 1, "replacement_text", 1.0, 3.0),
    ], 1.5)
    assert len([item for item in active if item["role"] == "replacement_text"]) == 2


def test_duplicate_ass_event_key_is_rejected():
    with pytest.raises(G514ValidationError, match="DUPLICATE_ASS_EVENT"):
        validate_dynamic_ass([
            _dialogue("e0", 1, "replacement_text"),
            _dialogue("e0", 1, "replacement_text"),
        ])


def test_global_fallback_is_rejected():
    with pytest.raises(G514ValidationError, match="GLOBAL_SUBTITLE_TRACK_NOT_DISABLED"):
        validate_dynamic_ass([_dialogue(None, 1, "replacement_text")])


def test_sequential_repeated_text_is_not_simultaneous_duplicate():
    dialogues = [
        _dialogue("e0", 1, "replacement_text", 0.0, 1.0),
        _dialogue("e1", 1, "replacement_text", 1.0, 2.0),
    ]
    assert [item["event_id"] for item in active_dialogues(dialogues, 0.5)] == ["e0"]
    assert [item["event_id"] for item in active_dialogues(dialogues, 1.5)] == ["e1"]


def test_timing_overlap_correction_is_deterministic():
    events = [
        {"start": 0.0, "end": 1.5},
        {"start": 1.0, "end": 2.0},
    ]
    assert correct_timing_boundaries(events) == correct_timing_boundaries(events)
    assert correct_timing_boundaries(events)[0]["end"] == 1.0


def test_timing_correction_rejects_zero_duration_event():
    with pytest.raises(G514ValidationError, match="ZERO_DURATION_EVENT"):
        correct_timing_boundaries([
            {"start": 0.0, "end": 1.0},
            {"start": 1.0, "end": 1.0},
        ])


@pytest.mark.parametrize("policy", [
    "REUSE_SOURCE_CONTAINER",
    "REPLACE_FULL_SOURCE_CONTAINER",
    "OPAQUE_TEXT_COVER",
])
def test_container_render_policy_is_parsed(policy):
    assert parse_container_policy({"recommended_render_policy": policy}) == policy


def test_visual_qa_false_positive_policy_and_hard_fails():
    assert validate_visual_result({
        "verdict": "PASS", "source_container_visible": True,
        "source_text_leakage": False, "duplicate_replacement": False,
        "replacement_text_overlap": False, "replacement_overflow": False,
        "unrelated_text_damage": False,
    })["status"] == "PASS"
    assert validate_visual_result({
        "verdict": "PASS", "source_text_leakage": False,
        "duplicate_replacement": True, "replacement_text_overlap": False,
        "replacement_overflow": False, "unrelated_text_damage": False,
    })["status"] == "FAIL"
