from src.application.services.subtitle_segmentation_service import SubtitleSegmentationService
from src.application.use_cases.subtitle_segmentation import SubtitleSegmentationUseCase
from src.domain.entities.segment import Segment
from src.domain.entities.subtitle_quality_config import SubtitleQualityConfig


def active_text(segment):
    return segment.translated_text or segment.text


def test_long_subtitle_splitting_preserves_text():
    cfg = SubtitleQualityConfig(max_chars_per_line=20, max_lines_per_subtitle=2, min_duration=0.8, max_duration=6.0)
    service = SubtitleSegmentationService(cfg)
    text = "This is a very long subtitle that must be split into multiple subtitle segments without losing any words."
    seg = Segment(id="s", start=0.0, end=8.0, text=text, source="speech", confidence=1.0)
    out = service.segment([seg])
    assert len(out) > 1
    reconstructed = " ".join(active_text(s).replace("\n", " ") for s in out)
    assert reconstructed == text


def test_proportional_timestamps_are_deterministic():
    cfg = SubtitleQualityConfig(max_chars_per_line=10, max_lines_per_subtitle=1, min_duration=0.8, max_duration=6.0)
    service = SubtitleSegmentationService(cfg)
    seg = Segment(id="s", start=0.0, end=8.0, text="one two three four five six seven eight", source="speech", confidence=1.0)
    first = service.segment([seg])
    seg2 = Segment(id="s", start=0.0, end=8.0, text="one two three four five six seven eight", source="speech", confidence=1.0)
    second = service.segment([seg2])
    assert [(s.start, s.end, s.text) for s in first] == [(s.start, s.end, s.text) for s in second]
    assert first[0].start == 0.0
    assert first[-1].end == 8.0


def test_duration_constraints():
    cfg = SubtitleQualityConfig(max_chars_per_line=8, max_lines_per_subtitle=1, min_duration=0.8, max_duration=3.0)
    service = SubtitleSegmentationService(cfg)
    seg = Segment(id="s", start=0.0, end=9.0, text="one two three four five six seven eight nine ten", source="speech", confidence=1.0)
    out = service.segment([seg])
    assert all((s.end - s.start) <= 3.0 for s in out[:-1])
    assert all((s.end - s.start) >= 0.8 for s in out[:-1])


def test_segmentation_use_case_pass_through_when_disabled():
    cfg = SubtitleQualityConfig(max_chars_per_line=5, max_lines_per_subtitle=1)
    uc = SubtitleSegmentationUseCase(SubtitleSegmentationService(cfg, enabled=False))
    seg = Segment(id="s", start=0.0, end=5.0, text="one two three four", source="speech", confidence=1.0)
    out = uc.execute([seg])
    assert out == [seg]
