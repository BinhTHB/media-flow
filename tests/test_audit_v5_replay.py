from scripts.audit_v5_replay import build_report, overlaps, primary_reason


def test_timestamp_mapping_uses_half_open_overlap():
    assert overlaps({"start": 0.0, "end": 3.3}, {"start": 3.0, "end": 4.0})
    assert not overlaps({"start": 0.0, "end": 3.0}, {"start": 3.0, "end": 4.0})


def test_primary_reason_is_deterministic_on_ties():
    decisions = [
        {"decision": "SKIP_REPLACEMENT", "reason": "insufficient_track_evidence"},
        {"decision": "SKIP_REPLACEMENT", "reason": "no_reliable_source_subtitle"},
    ]

    assert primary_reason(decisions) == "insufficient_track_evidence"


def test_report_maps_all_v5_captions_and_keeps_production_ready_no():
    v5 = {
        "captions": [
            {
                "caption_id": "v5_a",
                "timestamp_range": {"start": 0.0, "end": 1.5},
                "strategy": "expanded_line_bbox",
                "coordinate_mapping": "PASS",
            },
            {
                "caption_id": "v5_b",
                "timestamp_range": {"start": 1.5, "end": 2.0},
                "strategy": "expanded_line_bbox",
                "coordinate_mapping": "PASS",
            },
        ]
    }
    current = {
        "captions": [
            {
                "caption_id": "review_0",
                "timestamp_range": {"start": 0.0, "end": 1.0},
                "target_text": "target",
                "render_applied": False,
            },
            {
                "caption_id": "review_1",
                "timestamp_range": {"start": 1.0, "end": 2.0},
                "target_text": "target",
                "render_applied": True,
                "source_bbox": {"left": 1, "top": 1, "right": 2, "bottom": 2},
            },
        ]
    }
    gate = {
        "strict_policy_gate": {"decision": "BLOCK_SOURCE_REPLACE", "reason": "strict"},
        "adaptive_policy_gate": {"decision": "AUTO_SOURCE_REPLACE", "reason": None},
        "adaptive_decisions": [
            {
                "segment_id": "review_0",
                "decision": "SKIP_REPLACEMENT",
                "reason": "insufficient_track_evidence",
            },
            {
                "segment_id": "review_1",
                "decision": "AUTO_SOURCE_REPLACE",
                "reason": None,
            },
        ],
    }

    report = build_report(
        v5,
        current,
        gate,
        {"intervals": []},
        {"eligibility_decision": "SKIP_REPLACEMENT", "reasons": []},
        current,
    )

    assert report["counts"] == {
        "v5_caption_count": 2,
        "current_interval_count": 2,
        "current_adaptive_replacement_count": 1,
        "lost_count": 1,
        "mapped_v5_caption_count": 2,
    }
    assert len(report["caption_mappings"]) == 2
    assert report["Production Ready"] == "NO"
    assert report["verdicts"]["production_ready"] == "NO"
