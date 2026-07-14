from scripts.validate_output import normalized_text_preserved, srt_entries, timing_metrics


def test_timing_metrics_counts_short_and_overlapping_segments():
    metrics = timing_metrics(
        [
            {"start": 0.0, "end": 0.5},
            {"start": 0.3, "end": 1.3},
            {"start": 2.0, "end": 3.0},
        ]
    )

    assert metrics["duration_under_0_8s"] == 1
    assert metrics["overlap_count"] == 1
    assert metrics["gap_count"] == 1


def test_normalized_text_is_preserved_after_segmentation():
    normalized = [
        {
            "id": "speech_1",
            "translated_text": "Một câu\nhoàn chỉnh",
        }
    ]
    segmented = [
        {"id": "speech_1_0", "translated_text": "Một câu"},
        {"id": "speech_1_1", "translated_text": "hoàn chỉnh"},
    ]

    assert normalized_text_preserved(normalized, segmented)


def test_srt_entries_parse_text_and_timestamps(tmp_path):
    path = tmp_path / "subtitle.srt"
    path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nChào bạn\n\n",
        encoding="utf-8",
    )

    assert srt_entries(path) == [{"start": 0.0, "end": 1.0, "text": "Chào bạn"}]
