import copy
from pathlib import Path

import pytest

from scripts.run_g513_full_sample2 import (
    G513ValidationError,
    RENDER_NOT_APPLICABLE,
    build_quota_report,
    build_render_checkpoint,
    build_request_plan,
    build_v4_batch_checkpoint,
    select_eligible_render_events,
    source_text_by_event,
    validate_translation_checkpoint,
    validate_v4_batch_checkpoint,
    validate_v5_replay,
    validate_completed_stages,
    validate_no_completed_stage_requests,
    validate_render_checkpoint,
    validate_visual_qa_result,
    verify_legacy_id_mapping,
)


def _events(count=15):
    return [{
        "event_id": f"merged_0:e_{index}",
        "candidate_id": "merged_0",
        "start_seconds_in_source": float(index),
        "end_seconds_in_source": float(index) + 1.0,
        "representative_box_2d": [700, 100, 800, 900],
    } for index in range(count)]


def _replay(events):
    return {
        "video_sha256": "video",
        "v3_event_timing_sha256": __import__(
            "scripts.run_g513_full_sample2", fromlist=["event_timing_hash"]
        ).event_timing_hash(events),
        "provenance": [{
            "stage": "v5_transcription",
            "model": "gemini-3.5-flash-lite",
            "provider": "gemini_video_direct",
            "schema_version": "v1",
            "validator_version": "v1",
        }],
        "accepted_events": [{
            "event_id": event["event_id"],
            "text_verbatim": f"text {index}",
            "start_seconds_in_source": event["start_seconds_in_source"],
            "end_seconds_in_source": event["end_seconds_in_source"],
        } for index, event in enumerate(events)],
    }


def _manifest(events):
    return {
        "video_sha256": "video",
        "identity": {"video_sha256": "video", "model": "gemini-3.5-flash-lite"},
        "artifacts": {
            "stage_status": {
                stage: {"status": "COMPLETED_REUSED"}
                for stage in ("V1", "V2", "V3")
            },
            "checkpoints": {"temporal_events": {"events": events}},
        },
        "provenance": [],
    }


def test_g513_builds_fifteen_event_image_only_plan():
    events = _events()
    source_text = source_text_by_event(events, _replay(events))

    plan = build_request_plan(
        events, source_text, video_sha256="video", batch_size=5, rpm=15
    )

    assert plan["event_count"] == 15
    assert plan["generate_content_request_count"] == 3
    assert plan["count_tokens_call_count"] == 3
    assert plan["returned_token_count"] is None
    assert plan["direct_video_v4_requests"] == 0
    assert plan["minimum_request_spacing_seconds"] == 5.0
    assert all(item["frame_resolution"] == "original" for item in plan["events"])
    assert all(item["frame_mapping_sha256"] for item in plan["events"])


def test_g513_legacy_mapping_requires_complete_validated_ids():
    events = _events(2)
    replay = _replay(events)
    replay["accepted_events"] = [{
        **item, "event_id": item["event_id"].replace(":e_", ":")
    } for item in replay["accepted_events"]]
    assert verify_legacy_id_mapping(events, replay) == {
        "merged_0:0": "merged_0:e_0",
        "merged_0:1": "merged_0:e_1",
    }

    replay = _replay(events)
    replay["accepted_events"][1]["event_id"] = "merged_0:9"
    with pytest.raises(G513ValidationError, match="MAPPING"):
        verify_legacy_id_mapping(events, replay)


def test_g513_rejects_invalid_baseline_identity_and_timing():
    events = _events(1)
    manifest = _manifest(events)
    assert validate_completed_stages(manifest, "video") == events

    invalid_identity = copy.deepcopy(manifest)
    invalid_identity["identity"]["model"] = "other"
    with pytest.raises(G513ValidationError, match="IDENTITY"):
        validate_completed_stages(invalid_identity, "video")

    invalid_timing = copy.deepcopy(manifest)
    invalid_timing["artifacts"]["checkpoints"]["temporal_events"]["events"][0][
        "end_seconds_in_source"
    ] = 0.0
    with pytest.raises(G513ValidationError, match="TIMING"):
        validate_completed_stages(invalid_timing, "video")


def test_g513_rejects_completed_stage_request():
    with pytest.raises(G513ValidationError, match="UNEXPECTED_COMPLETED_STAGE_REQUEST"):
        validate_no_completed_stage_requests(
            {"completed_stage_request_count": 0},
            [{"stage": "v3_segmentation"}],
        )


def test_g513_eligible_render_selection_accepts_wide_bounded_box():
    events = _events(1)
    geometry = [{
        "event_id": "merged_0:e_0",
        "representative_box_2d": [700, 10, 800, 990],
        "needs_review": False,
    }]
    translations = [{"event_id": "merged_0:e_0", "translated_text": "Bản dịch"}]

    selected = select_eligible_render_events(events, geometry, translations)

    assert [item["event_id"] for item in selected] == ["merged_0:e_0"]
    assert selected[0]["representative_box_2d"] == [700, 10, 800, 990]


def test_g513_render_checkpoint_reuse_invalidates_hash_mismatch(tmp_path):
    input_video = tmp_path / "input.mp4"
    output_video = tmp_path / "output.mp4"
    ass_path = tmp_path / "subtitles.ass"
    input_video.write_bytes(b"input")
    output_video.write_bytes(b"output")
    ass_path.write_text("ass", encoding="utf-8")
    geometry = [{"event_id": "e", "representative_box_2d": [1, 2, 3, 4]}]
    translations = [{"event_id": "e", "translated_text": "text"}]
    checkpoint = build_render_checkpoint(
        input_video, output_video, geometry, translations, ass_path, 0
    )

    assert checkpoint["request_status"] == RENDER_NOT_APPLICABLE
    assert validate_render_checkpoint(
        checkpoint, input_video, output_video, geometry, translations, ass_path
    )
    output_video.write_bytes(b"changed")
    assert not validate_render_checkpoint(
        checkpoint, input_video, output_video, geometry, translations, ass_path
    )


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"status": "PASS", "source_text_visible": False}, "PASS"),
        ({"status": "PASS", "source_text_visible": True}, "FAIL"),
        ({"status": "REVIEW_REQUIRED", "source_text_visible": False}, "REVIEW_REQUIRED"),
        ({"status": "unknown"}, "REVIEW_REQUIRED"),
    ],
)
def test_g513_visual_qa_rejects_leakage_and_defers_unclear_output(result, expected):
    assert validate_visual_qa_result(result)["status"] == expected

def test_g513_v5_replay_fails_closed_without_full_identity_proof():
    events = _events()
    assert validate_v5_replay(events, _replay(events), "video")["merged_0:e_0"] == "text 0"

    replay = _replay(events)
    del replay["accepted_events"][0]["start_seconds_in_source"]
    with pytest.raises(G513ValidationError, match="V5_CHECKPOINT_IDENTITY_BLOCKED"):
        validate_v5_replay(events, replay, "video")

    replay = _replay(events)
    replay["provenance"][0]["validator_version"] = ""
    with pytest.raises(G513ValidationError, match="V5_CHECKPOINT_IDENTITY_BLOCKED"):
        validate_v5_replay(events, replay, "video")

def test_g513_translation_checkpoint_reuses_only_complete_provenanced_data():
    events = _events(2)
    checkpoint = {
        "input_v5_sha256": "v5",
        "provenance": [{
            "stage": "translation", "model": "gemini-3.5-flash-lite",
            "provider": "gemini_video_direct", "schema_version": "v1",
            "validator_version": "v1",
        }],
        "events": [
            {"event_id": event["event_id"], "translated_text": f"vi {index}"}
            for index, event in enumerate(events)
        ],
    }
    reused, rerun = validate_translation_checkpoint(checkpoint, events, "v5")
    assert len(reused) == 2
    assert rerun == []

    checkpoint["events"][1]["translated_text"] = ""
    reused, rerun = validate_translation_checkpoint(checkpoint, events, "v5")
    assert reused == []
    assert rerun == [event["event_id"] for event in events]

def test_g513_v4_batch_checkpoint_and_quota_report_are_hash_bound():
    batch = _events(2)
    geometry = [{"event_id": item["event_id"], "needs_review": False} for item in batch]
    checkpoint = build_v4_batch_checkpoint(
        batch, geometry, "video", [{"stage": "v4_image_geometry"}]
    )
    assert validate_v4_batch_checkpoint(checkpoint, batch, "video")
    checkpoint["geometry"][0]["needs_review"] = True
    assert not validate_v4_batch_checkpoint(checkpoint, batch, "video")

    report = build_quota_report([
        {"count_tokens": 7, "actual_usage": {"total_token_count": 12}},
        {"count_tokens": 9, "actual_usage": {"totalTokenCount": 8}, "status_code": 429},
    ], historical_429=3)
    assert report == {
        "generate_content_request_count": 2,
        "count_tokens_call_count": 2,
        "returned_token_count": 20,
        "current_run_429_count": 1,
        "historical_429_count": 3,
    }
