from scripts.replay_g58_v5_validation import replay


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
