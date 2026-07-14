from scripts.benchmark_asr import (
    error_rates,
    homophone_analysis,
    parse_substitutions,
    recommendation,
    segment_metrics,
)


def test_segment_metrics_counts_core_values():
    metrics = segment_metrics(
        [
            {"start": 0.0, "end": 1.0, "text": "你好 世界", "confidence": 0.8},
            {"start": 1.0, "end": 3.0, "text": "你好 世界", "confidence": 0.6},
            {"start": 3.0, "end": 3.5, "text": "", "confidence": 0.0},
        ]
    )

    assert metrics["segment_count"] == 3
    assert metrics["character_count"] == 8
    assert metrics["word_count"] == 4
    assert metrics["empty_segment_count"] == 1
    assert metrics["duplicate_segment_count"] == 1
    assert metrics["average_confidence"] == 0.7


def test_error_rates_skip_missing_reference_and_calculate_cer():
    assert error_rates("你好", None) is None
    assert error_rates("你好", "你好") == {"cer": 0.0, "wer": 0.0}
    assert error_rates("你号", "你好")["cer"] == 0.5


def test_homophone_analysis_reports_affected_timestamps():
    findings = homophone_analysis(
        [
            {"id": "speech_1", "start": 1.0, "end": 2.0, "text": "英国报应"},
            {"id": "speech_2", "start": 2.0, "end": 3.0, "text": "因果循环"},
        ],
        [("因果", "英国")],
    )

    assert findings[0]["frequency"] == 1
    assert findings[0]["affected_timestamps"][0]["id"] == "speech_1"


def test_recommendation_does_not_change_production_defaults():
    assert recommendation(["tiny", "base", "small"]) == {
        "smoke_test": "tiny",
        "development": "base",
        "production": "small",
        "chinese_short_videos": "small",
        "english_short_videos": "base",
    }


def test_parse_substitutions_validates_input():
    assert parse_substitutions(["因果:英国"]) == [("因果", "英国")]
