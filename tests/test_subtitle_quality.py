import pytest
from src.domain.entities.segment import Segment
from src.domain.entities.subtitle_quality_config import SubtitleQualityConfig
from src.application.use_cases.normalize_segments import NormalizeSegmentsUseCase
from src.application.services.text_similarity import TextSimilarityService
from src.application.services.subtitle_text_formatter import SubtitleTextFormatter
from src.application.services.speech_quality_auditor import SpeechQualityAuditor
from src.application.services.vietnamese_text_quality_auditor import (
    VietnameseTextQualityAuditor,
)
from src.shared.exceptions import ValidationError

def test_subtitle_quality_config_validation():
    # Valid
    cfg = SubtitleQualityConfig()
    assert cfg.max_chars_per_line == 42
    
    # Invalid max_chars_per_line
    with pytest.raises(ValidationError):
        SubtitleQualityConfig(max_chars_per_line=0)
        
    # Invalid max_lines_per_subtitle
    with pytest.raises(ValidationError):
        SubtitleQualityConfig(max_lines_per_subtitle=0)

    # Invalid min_duration
    with pytest.raises(ValidationError):
        SubtitleQualityConfig(min_duration=-1.0)
        
    # Invalid max_duration < min_duration
    with pytest.raises(ValidationError):
        SubtitleQualityConfig(min_duration=3.0, max_duration=2.0)

def test_text_similarity_service():
    sim = TextSimilarityService.get_similarity("Hello World", "hello world")
    assert sim == 1.0
    
    sim = TextSimilarityService.get_similarity("Hello World 1", "Hello World 2")
    assert sim > 0.8

def test_subtitle_text_formatter():
    text = "Chào mọi người, chào mừng đến với kênh của tôi."
    formatted = SubtitleTextFormatter.format_text(text, max_chars_per_line=20)
    lines = formatted.split("\n")
    assert len(lines) == 3
    assert len(lines[0]) <= 20
    assert len(lines[1]) <= 20
    assert len(lines[2]) <= 20

def test_normalize_empty_and_whitespace():
    uc = NormalizeSegmentsUseCase(SubtitleQualityConfig(
        remove_empty_segments=True,
        trim_whitespace=True
    ))
    segments = [
        Segment(id="1", start=0, end=1.5, text="   hello   ", source="speech"),
        Segment(id="2", start=2, end=3.0, text="   ", source="speech"),
    ]
    res = uc.execute(segments, is_translated=False)
    assert len(res) == 1
    assert res[0].text == "hello"

def test_normalize_sorting():
    uc = NormalizeSegmentsUseCase()
    segments = [
        Segment(id="1", start=2.0, end=3.0, text="second", source="speech"),
        Segment(id="2", start=1.0, end=1.5, text="first", source="speech"),
    ]
    res = uc.execute(segments)
    assert res[0].id == "2"
    assert res[1].id == "1"

def test_normalize_prefer_speech_over_ocr():
    uc = NormalizeSegmentsUseCase(SubtitleQualityConfig(
        prefer_speech_over_ocr=True
    ))
    segments = [
        Segment(id="s1", start=0.0, end=3.0, text="Chào mọi người", source="speech"),
        Segment(id="o1", start=0.5, end=2.5, text="Chào mọi người", source="ocr"),
    ]
    res = uc.execute(segments)
    assert len(res) == 1
    assert res[0].source == "speech"

def test_normalize_deduplicate():
    uc = NormalizeSegmentsUseCase(SubtitleQualityConfig(
        deduplicate=True
    ))
    segments = [
        Segment(id="s1", start=0.0, end=3.0, text="Chào thế giới", source="speech"),
        Segment(id="s2", start=3.5, end=6.5, text="Chào thế giới", source="speech"),
    ]
    res = uc.execute(segments)
    assert len(res) == 1
    assert res[0].id == "s1"

def test_normalize_merge_short_segments():
    uc = NormalizeSegmentsUseCase(SubtitleQualityConfig(
        merge_gap_threshold=0.5,
        max_duration=6.0
    ))
    segments = [
        Segment(id="s1", start=0.0, end=1.5, text="Chào", source="speech"),
        Segment(id="s2", start=1.8, end=3.0, text="bạn", source="speech"),
    ]
    res = uc.execute(segments)
    assert len(res) == 1
    assert res[0].text == "Chào bạn"
    assert res[0].start == 0.0
    assert res[0].end == 3.0

def test_normalize_clamp_duration():
    uc = NormalizeSegmentsUseCase(SubtitleQualityConfig(
        min_duration=1.0,
        max_duration=4.0
    ))
    segments = [
        # Duration 0.5 -> clamp to 1.0
        Segment(id="1", start=0.0, end=0.5, text="short", source="speech"),
        # Duration 5.0 -> clamp to 4.0
        Segment(id="2", start=1.0, end=6.0, text="long", source="speech"),
    ]
    res = uc.execute(segments)
    assert res[0].end - res[0].start == 1.0
    assert res[1].end - res[1].start == 4.0


def test_normalize_keeps_best_overlapping_caption():
    uc = NormalizeSegmentsUseCase()
    segments = [
        Segment(id="low", start=0.0, end=3.0, text="bad", source="ocr", confidence=0.2),
        Segment(id="high", start=1.0, end=4.0, text="clear caption", source="speech", confidence=0.9),
    ]
    result = uc.execute(segments, video_duration=10.0)
    assert [segment.id for segment in result] == ["low", "high"]
    assert all(segment.start < segment.end for segment in result)
    assert result[0].end <= result[1].start
    assert all(segment.start < segment.end for segment in result)


def test_normalize_clamps_captions_to_video_duration():
    uc = NormalizeSegmentsUseCase()
    result = uc.execute(
        [Segment(id="late", start=9.5, end=12.0, text="caption", source="speech")],
        video_duration=10.0,
    )
    assert result == []


def test_normalize_drops_zero_duration_captions():
    uc = NormalizeSegmentsUseCase()
    result = uc.execute(
        [Segment(id="zero", start=2.0, end=2.0, text="caption", source="speech")],
        video_duration=10.0,
    )
    assert result == []


def test_normalize_resolves_nested_overlaps():
    uc = NormalizeSegmentsUseCase()
    segments = [
        Segment(id="outer", start=0.0, end=6.0, text="outer caption", source="speech", confidence=0.6),
        Segment(id="nested", start=1.0, end=2.0, text="better", source="ocr", confidence=0.9),
    ]
    result = uc.execute(segments, video_duration=10.0)
    assert [segment.id for segment in result] == ["nested"]


def test_normalize_allows_only_one_simultaneous_caption():
    uc = NormalizeSegmentsUseCase()
    segments = [
        Segment(id="one", start=0.0, end=3.0, text="one", source="speech", confidence=0.1),
        Segment(id="two", start=0.5, end=3.5, text="two", source="speech", confidence=0.2),
        Segment(id="three", start=1.0, end=4.0, text="three", source="speech", confidence=0.3),
    ]
    result = uc.execute(segments, video_duration=10.0)
    assert [segment.id for segment in result] == ["three"]

def test_normalize_queues_long_nonduplicate_speech_without_overlaps():
    uc = NormalizeSegmentsUseCase()
    segments = [
        Segment(id="speech", start=0.0, end=12.0, text="spoken commentary", source="speech"),
        Segment(id="ocr1", start=2.0, end=4.0, text="first label", source="ocr", confidence=0.9),
        Segment(id="ocr2", start=6.0, end=8.0, text="second label", source="ocr", confidence=0.9),
        Segment(id="ocr3", start=10.0, end=12.0, text="third label", source="ocr", confidence=0.9),
    ]

    result = uc.execute(segments, video_duration=12.0)

    assert sum(segment.source == "speech" for segment in result) == 2
    assert {segment.id for segment in result if segment.source == "ocr"} == {
        "ocr1", "ocr2", "ocr3"
    }
    assert " ".join(
        segment.text for segment in result if segment.source == "speech"
    ) == "spoken commentary"
    assert uc.metrics["long_speech_segments_detected"] == 1
    assert uc.metrics["long_speech_segments_split_or_queued"] == 1
    assert uc.metrics["long_speech_segments_dropped"] == 0
    assert uc.metrics["speech_coverage_ratio"] > 0
    assert all(0 <= segment.start < segment.end <= 12.0 for segment in result)
    assert all(left.end <= right.start for left, right in zip(result, result[1:]))

def test_normalize_long_speech_duplicate_with_ocr_is_deduplicated():
    uc = NormalizeSegmentsUseCase()
    result = uc.execute(
        [
            Segment(id="speech", start=0.0, end=10.0, text="same caption", source="speech"),
            Segment(id="ocr1", start=1.0, end=3.0, text="same caption", source="ocr"),
            Segment(id="ocr2", start=5.0, end=7.0, text="same caption", source="ocr"),
        ],
        video_duration=10.0,
    )

    assert [segment.id for segment in result] == ["speech"]
    assert uc.metrics["long_speech_segments_detected"] == 1
    assert uc.metrics["long_speech_segments_split_or_queued"] == 0
    assert all(segment.start < segment.end for segment in result)

def test_speech_quality_report_flags_low_information_asr_output():
    asr_segments = [
        Segment(
            id="speech_0",
            start=0.0,
            end=24.3,
            text="遇到了",
            source="speech",
            confidence=0.0,
        )
    ]
    report = SpeechQualityAuditor.report(
        asr_segments,
        asr_segments,
        [
            Segment(
                id="speech_0",
                start=0.0,
                end=24.3,
                text="遇到了",
                translated_text="đã gặp",
                source="speech",
                confidence=0.0,
            )
        ],
        "zh",
    )

    assert report["asr_segment_count"] == 1
    assert report["speech_translation_input_count"] == 1
    assert report["speech_translation_output_count"] == 1
    assert report["speech_collapse_detected"] is True
    assert "low_information_asr_output" in report["warnings"]
    assert "low_confidence_asr_output" in report["warnings"]
    assert "zero_confidence_asr_output" in report["warnings"]
    assert "single_long_segment_with_tiny_text" in report["warnings"]
    assert "very_low_chars_per_second" in report["warnings"]

def test_speech_quality_report_records_translation_count_collapse():
    report = SpeechQualityAuditor.report(
        [
            Segment(id="speech_0", start=0.0, end=1.0, text="one", source="speech"),
            Segment(id="speech_1", start=2.0, end=3.0, text="two", source="speech"),
        ],
        [
            Segment(id="speech_0", start=0.0, end=1.0, text="one", source="speech"),
            Segment(id="speech_1", start=2.0, end=3.0, text="two", source="speech"),
        ],
        [Segment(id="speech_0", start=0.0, end=3.0, text="one two", source="speech")],
        "en",
    )

    assert report["speech_collapse_detected"] is True
    assert "speech_segment_count_collapsed" in report["warnings"]

def test_speech_quality_report_accepts_informative_asr_output():
    segments = [
        Segment(
            id=f"speech_{index}",
            start=float(index * 2),
            end=float(index * 2 + 2),
            text=text,
            translated_text=text,
            source="speech",
            confidence=0.85,
        )
        for index, text in enumerate(
            ["这是清晰的第一句话", "这里还有更多语音内容", "最后一句也可以正常识别"]
        )
    ]

    report = SpeechQualityAuditor.report(segments, segments, segments, "zh")

    assert report["speech_collapse_detected"] is False
    assert report["asr_chars_per_second"] >= 0.5
    assert report["asr_low_confidence_count"] == 0
    assert report["warnings"] == []

def test_speech_quality_report_flags_language_mismatch():
    segment = Segment(
        id="speech_0",
        start=0.0,
        end=2.0,
        text="Nội dung tiếng Việt rõ ràng",
        translated_text="Nội dung tiếng Việt rõ ràng",
        source="speech",
        confidence=0.9,
    )

    report = SpeechQualityAuditor.report(
        [segment], [segment], [segment], "zh", detected_language="vi"
    )

    assert "language_mismatch" in report["warnings"]

def test_normalize_suppresses_redundant_incomplete_ocr_fragment():
    uc = NormalizeSegmentsUseCase()
    result = uc.execute(
        [
            Segment(
                id="speech",
                start=0.0,
                end=3.0,
                text="Ngay lúc anh ta tưởng rằng chiếc xe sẽ bị tịch thu.",
                source="speech",
                confidence=0.8,
            ),
            Segment(
                id="ocr",
                start=1.0,
                end=2.0,
                text="ngay lúc anh",
                source="ocr",
                confidence=0.99,
            ),
        ],
        video_duration=4.0,
    )

    assert [segment.id for segment in result] == ["speech"]
    assert uc.metrics["redundant_ocr_count"] == 1
    assert uc.metrics["suppressed_ocr_fragments"][0]["id"] == "ocr"
    assert uc.metrics["dropped_speech_segments"] == []

def test_normalize_preserves_useful_ocr_not_covered_by_speech():
    uc = NormalizeSegmentsUseCase()
    result = uc.execute(
        [
            Segment(
                id="speech",
                start=0.0,
                end=2.0,
                text="Anh ta đang lái xe trên đường.",
                source="speech",
                confidence=0.8,
            ),
            Segment(
                id="ocr",
                start=2.2,
                end=3.5,
                text="CẢNH BÁO ĐƯỜNG CẤM",
                source="ocr",
                confidence=0.99,
            ),
        ],
        video_duration=4.0,
    )

    assert {segment.id for segment in result} == {"speech", "ocr"}
    assert uc.metrics["redundant_ocr_count"] == 0

def test_normalize_reports_incomplete_ocr_without_speech_coverage():
    uc = NormalizeSegmentsUseCase()
    result = uc.execute(
        [Segment(id="ocr", start=0.0, end=2.0, text="sau đó tuyên", source="ocr")],
        video_duration=3.0,
    )

    assert [segment.id for segment in result] == ["ocr"]
    assert uc.metrics["suppressed_ocr_fragments"] == []

def test_normalize_fusion_timeline_remains_non_overlapping():
    uc = NormalizeSegmentsUseCase(SubtitleQualityConfig(merge_gap_threshold=0.0))
    result = uc.execute(
        [
            Segment(id="speech0", start=0.0, end=3.0, text="Câu nói đầu tiên đầy đủ.", source="speech"),
            Segment(id="ocr0", start=1.0, end=2.0, text="câu nói", source="ocr"),
            Segment(id="speech1", start=3.0, end=5.0, text="Nội dung kết thúc hoàn toàn khác biệt.", source="speech"),
        ],
        video_duration=5.0,
    )

    assert all(left.end <= right.start for left, right in zip(result, result[1:]))
    assert all(segment.start < segment.end for segment in result)
    assert uc.metrics["final_speech_caption_count"] == 2
    assert uc.metrics["final_ocr_caption_count"] == 0

@pytest.mark.parametrize(
    ("incorrect", "corrected"),
    [
        ("Viên cảnh xác đang đứng.", "Viên cảnh sát đang đứng."),
        ("Anh ta bắt ngờ dừng lại.", "Anh ta bất ngờ dừng lại."),
        ("Người này có vẽ mặc nghiêm túc.", "Người này có vẻ mặt nghiêm túc."),
    ],
)
def test_vietnamese_text_quality_flags_high_confidence_typos(incorrect, corrected):
    report = VietnameseTextQualityAuditor.audit(incorrect)

    assert report["suggested_text"] == corrected
    assert report["corrections"][0]["confidence"] == 0.99
    assert report["actual_action"] == "flag"

def test_vietnamese_text_quality_keeps_clean_caption():
    report = VietnameseTextQualityAuditor.audit(
        "Viên cảnh sát bất ngờ ngồi xuống với vẻ mặt nghiêm túc."
    )

    assert report["corrections"] == []
    assert report["suggested_text"] == report["original_text"]
    assert report["actual_action"] == "keep"

def test_vietnamese_text_quality_report_only_does_not_change_text():
    text = "Viên cảnh xác bắt ngờ."
    report = VietnameseTextQualityAuditor.apply(text, enabled=False)

    assert report["corrected_text"] == text
    assert report["actual_action"] == "flag"

def test_vietnamese_text_quality_correction_mode_applies_suggestion():
    report = VietnameseTextQualityAuditor.apply(
        "Viên cảnh xác bắt ngờ.", enabled=True
    )

    assert report["corrected_text"] == "Viên cảnh sát bất ngờ."
    assert report["original_text"] == "Viên cảnh xác bắt ngờ."
    assert report["actual_action"] == "correct"
