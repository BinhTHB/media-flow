from scripts.replay_g58_v5_validation import apply_checkpoint, replay


def test_offline_replay_accepts_persisted_v5_payload_without_network():
    manifest = {
        "artifacts": {
            "checkpoints": {
                "temporal_events": {
                    "events": [{"event_id": "event-1"}],
                },
                "v5_raw_transcription": {
                    "batches": [{
                        "payload": {
                            "results": [{
                                "event_id": "event-1",
                                "text_verbatim": "你好",
                                "language": "zh",
                                "script": "Hans",
                                "status": "ok",
                            }],
                        },
                    }],
                },
            },
        },
    }

    report = replay(manifest, "zh")

    assert report["verdict"] == "V5_VALIDATION_REPLAY_PASS"
    assert report["network_requests"] == 0
    assert report["count_tokens_calls"] == 0
    assert report["files_api_calls"] == 0
    assert report["accepted_events"][0]["text_verbatim"] == "你好"


def test_offline_replay_reports_missing_and_duplicate_event_ids():
    manifest = {
        "artifacts": {
            "checkpoints": {
                "temporal_events": {
                    "events": [{"event_id": "event-1"}, {"event_id": "event-2"}],
                },
                "v5_raw_transcription": {
                    "batches": [{
                        "payload": {
                            "results": [
                                {"event_id": "event-1", "text_verbatim": "one", "status": "success"},
                                {"event_id": "event-1", "text_verbatim": "duplicate", "status": "success"},
                            ],
                        },
                    }],
                },
            },
        },
    }

    report = replay(manifest)

    assert report["duplicate_event_ids"] == ["event-1"]
    assert report["missing_event_ids"] == ["event-2"]
    assert report["rejected_events"][0]["reject_reason"] == "MISSING_EVENT_RESULT"


def test_apply_checkpoint_separates_request_and_validation_status():
    manifest = {
        "artifacts": {"checkpoints": {
            "temporal_events": {"events": [{"event_id": "e", "candidate_id": "c"}]},
            "geometry": {"events": [{"event_id": "e", "representative_box_2d": [1, 1, 2, 2]}]},
        }},
    }
    report = {
        "verdict": "V5_VALIDATION_REPLAY_PASS",
        "accepted_events": [{"event_id": "e", "text_verbatim": "Xin chào",
                             "language": "vi", "script": "Latn", "status": "readable"}],
        "raw_artifact_hash": "raw", "validator_version": "test",
    }

    apply_checkpoint(manifest, report)

    checkpoint = manifest["artifacts"]["checkpoints"]["transcriptions"]
    assert checkpoint["request_status"] == "COMPLETED_REUSED"
    assert checkpoint["validation_status"] == "PASS"
    assert checkpoint["stage_status"] == "COMPLETED"
