import pytest
from src.domain.entities.segment import Segment
from src.domain.entities.subtitle_quality_config import SubtitleQualityConfig
from src.application.use_cases.filter_ocr_segments import FilterOCRSegmentsUseCase
from src.application.use_cases.group_segments import GroupSegmentsUseCase
from src.application.services.source_text_leakage_detector import SourceTextLeakageDetector
from src.shared.exceptions import ValidationError
from src.application.services.content_quality_auditor import ContentQualityAuditor


def test_content_quality_auditor_findings():
    assert ContentQualityAuditor.audit("...", "speech") == (0.0, ["placeholder_only"])
    assert "malformed_ocr_spelling" in ContentQualityAuditor.audit("tô điu khin", "ocr")[1]
    assert "incomplete_vietnamese_fragment" in ContentQualityAuditor.audit("bảo hiểm cho", "ocr")[1]
    assert ContentQualityAuditor.audit("viên cảnh sát", "ocr") == (1.0, [])

def test_filter_ocr_low_confidence():
    cfg = SubtitleQualityConfig(ocr_min_confidence=0.80)
    uc = FilterOCRSegmentsUseCase(cfg)
    segments = [
        Segment(id="ocr_1", start=1.0, end=3.0, text="Valid OCR", source="ocr", confidence=0.85),
        Segment(id="ocr_2", start=2.0, end=4.0, text="Low Conf OCR", source="ocr", confidence=0.75),
    ]
    res, metrics = uc.execute(segments, source_language="zh")
    assert len(res) == 1
    assert res[0].id == "ocr_1"
    assert metrics["dropped_by_confidence"] == 1

def test_filter_ocr_single_character_noise():
    cfg = SubtitleQualityConfig()
    uc = FilterOCRSegmentsUseCase(cfg)
    segments = [
        # Allowed Chinese char in zh
        Segment(id="ocr_1", start=1.0, end=3.0, text="日", source="ocr", confidence=0.90),
        # Allowed digit
        Segment(id="ocr_2", start=2.0, end=4.0, text="5", source="ocr", confidence=0.90),
        # Allowed single letters
        Segment(id="ocr_3", start=3.0, end=5.0, text="a", source="ocr", confidence=0.90),
        # Noise character to drop
        Segment(id="ocr_4", start=4.0, end=6.0, text="x", source="ocr", confidence=0.90),
    ]
    res, metrics = uc.execute(segments, source_language="zh")
    ids = {s.id for s in res}
    assert "ocr_4" not in ids
    assert "ocr_1" in ids
    assert "ocr_2" in ids
    assert "ocr_3" in ids
    assert metrics["dropped_by_length"] == 1

def test_filter_ocr_acronym_and_number_preservation():
    cfg = SubtitleQualityConfig(ocr_max_abnormal_char_ratio=0.1)
    uc = FilterOCRSegmentsUseCase(cfg)
    segments = [
        # Acronym (all caps, length >= 2)
        Segment(id="ocr_1", start=1.0, end=3.0, text="MVP", source="ocr", confidence=0.90),
        # Number
        Segment(id="ocr_2", start=2.0, end=4.0, text="1105", source="ocr", confidence=0.90),
        # Text with symbols that should be preserved because it's clean
        Segment(id="ocr_3", start=3.0, end=5.0, text="Clean text.", source="ocr", confidence=0.90),
    ]
    res, metrics = uc.execute(segments, source_language="zh")
    assert len(res) == 3
    assert metrics["dropped_by_abnormal_characters"] == 0

def test_filter_ocr_abnormal_symbol_ratio():
    cfg = SubtitleQualityConfig(ocr_max_abnormal_char_ratio=0.2)
    uc = FilterOCRSegmentsUseCase(cfg)
    segments = [
        # Normal text with punctuation (ratio = 1/11 = 0.09 < 0.2)
        Segment(id="ocr_1", start=1.0, end=3.0, text="Hello world!", source="ocr", confidence=0.90),
        # Abnormal text with weird symbols (e.g. 3 weird symbols out of 7 characters = 0.42 > 0.2)
        Segment(id="ocr_2", start=2.0, end=4.0, text="A#$*B@1", source="ocr", confidence=0.90),
    ]
    res, metrics = uc.execute(segments, source_language="zh")
    assert len(res) == 1
    assert res[0].id == "ocr_1"
    assert metrics["dropped_by_abnormal_characters"] == 1

def test_filter_ocr_repetition():
    cfg = SubtitleQualityConfig(ocr_repetition_window=2.0)
    uc = FilterOCRSegmentsUseCase(cfg)
    segments = [
        Segment(id="ocr_1", start=1.0, end=3.0, text="Duplicate text", source="ocr", confidence=0.90),
        # Repeated within window (1.8s - 1.0s = 0.8s <= 2.0s)
        Segment(id="ocr_2", start=1.8, end=3.8, text="Duplicate text", source="ocr", confidence=0.90),
        # Repeated outside window (4.0s - 1.0s = 3.0s > 2.0s)
        Segment(id="ocr_3", start=4.0, end=6.0, text="Duplicate text", source="ocr", confidence=0.90),
    ]
    res, metrics = uc.execute(segments, source_language="zh")
    assert len(res) == 2
    assert res[0].id == "ocr_1"
    assert res[1].id == "ocr_3"
    assert metrics["dropped_by_repetition"] == 1

def test_source_text_leakage_detection():
    # Chinese source language leakage
    zh_segments = [
        Segment(id="1", start=0.0, end=1.0, text="你好", translated_text="你好", source="speech"),
        Segment(id="2", start=1.0, end=2.0, text="你好", translated_text="Xin chào", source="speech"),
    ]
    assert SourceTextLeakageDetector.count(zh_segments, source_language="zh") == 1

    # English/general source language leakage (identical text, not acronym or number)
    en_segments = [
        Segment(id="1", start=0.0, end=1.0, text="Hello", translated_text="Hello", source="speech"),
        # Number: not a leak
        Segment(id="2", start=1.0, end=2.0, text="12345", translated_text="12345", source="speech"),
        # Acronym: not a leak
        Segment(id="3", start=2.0, end=3.0, text="API", translated_text="API", source="speech"),
        # Translated: not a leak
        Segment(id="4", start=3.0, end=4.0, text="Hello", translated_text="Xin chào", source="speech"),
    ]
    assert SourceTextLeakageDetector.count(en_segments, source_language="en") == 1

def test_vietnamese_ocr_is_not_source_leakage():
    segments = [
        Segment(
            id="ocr_vi",
            start=3.5,
            end=5.5,
            text="anh ta không",
            translated_text="anh ta không",
            source="ocr",
        )
    ]
    assert SourceTextLeakageDetector.detect(segments, source_language="zh") == []

def test_source_text_leakage_examples_include_context():
    segment = Segment(
        id="leak",
        start=1.25,
        end=2.5,
        text="你好",
        translated_text="你好",
        source="ocr",
    )
    examples = SourceTextLeakageDetector.detect([segment], source_language="zh")
    assert examples == [{
        "index": 1,
        "timestamp": "1.250 --> 2.500",
        "text": "你好",
        "translated_text": "你好",
        "source": "ocr",
        "reason": "Chinese characters remaining in translation",
    }]

def test_group_segments():
    cfg = SubtitleQualityConfig(ocr_grouping_max_gap=1.5)
    uc = GroupSegmentsUseCase(cfg)
    segments = [
        # Group 1: OCR clean segments
        Segment(id="ocr_1", start=1.0, end=3.0, text="Hello", source="ocr"),
        Segment(id="ocr_2", start=4.0, end=5.0, text="world", source="ocr"),
        
        # Segment 3: ending punctuation (do not merge next)
        Segment(id="ocr_3", start=6.0, end=7.0, text="done.", source="ocr"),
        Segment(id="ocr_4", start=7.2, end=8.0, text="Next", source="ocr"),
        
        # Segment 5: different source type (do not merge with OCR)
        Segment(id="speech_1", start=8.2, end=9.5, text="speech", source="speech"),
    ]
    
    grouped = uc.execute(segments, source_language="en")
    
    # Expected groups:
    # 1. "Hello" + "world" + "done." -> "Hello world done." (1.0 to 7.0)
    # 2. "Next" (7.2 to 8.0)
    # 3. "speech" (8.2 to 9.5)
    assert len(grouped) == 3
    assert grouped[0].text == "Hello world done."
    assert grouped[0].start == 1.0
    assert grouped[0].end == 7.0
    assert grouped[1].text == "Next"
    assert grouped[2].text == "speech"

def test_chinese_group_segments_no_spaces():
    cfg = SubtitleQualityConfig(ocr_grouping_max_gap=1.5)
    uc = GroupSegmentsUseCase(cfg)
    segments = [
        Segment(id="ocr_1", start=1.0, end=3.0, text="你好", source="ocr"),
        Segment(id="ocr_2", start=4.0, end=5.0, text="世界", source="ocr"),
    ]
    grouped = uc.execute(segments, source_language="zh")
    assert len(grouped) == 1
    assert grouped[0].text == "你好世界" # No spaces for Chinese characters
