"""Reup quality: keep captions that are already Vietnamese, drop OCR counters,
and do not dub a line the translator echoed in the source language."""

from src.core.models.bbox import BBox
from src.core.models.speech_segment import SpeechSegment
from src.core.models.subtitle_event import SubtitleEvent
from src.features.audio.models import AudioRequest
from src.features.audio.pipeline import AudioPipeline
from src.features.translation.text_quality import (
    accept_spoken_line,
    is_already_target_language,
    is_caption_worthy,
    is_effectively_identical,
)
from src.features.visual.subtitle.processor import SubtitleProcessor


def test_caption_and_language_gates():
    assert is_already_target_language("Kì lạ thật", "vi")
    assert not is_already_target_language("Hello there", "vi")
    assert not is_already_target_language("你好", "vi")
    assert not is_caption_worthy("122")
    assert not is_caption_worthy("ab 1234")
    assert is_caption_worthy("你好")
    assert is_effectively_identical("Kỳ lạ!", "ky la")
    assert accept_spoken_line("你好", "你好", "vi") == ""
    assert accept_spoken_line("你好", "Xin chào", "vi") == "Xin chào"
    assert accept_spoken_line("Kì lạ thật", "", "vi") == "Kì lạ thật"
    assert accept_spoken_line("Kì lạ thật", "Lạ thật", "vi") == "Lạ thật"


class _Segmenter:
    def __init__(self, events):
        self.events = events

    def segment_track(self, track, video_path=None):
        return list(self.events)


class _Translator:
    def __init__(self, mapping):
        self.mapping = mapping
        self.seen = []

    def translate_texts(self, texts, target_language="vi", source_language="auto"):
        self.seen.extend(texts)
        return [self.mapping.get(text, f"Dịch {text}") for text in texts]


def _caption(event_id, text, start):
    return SubtitleEvent(
        event_id=event_id,
        track_id="T",
        start_time=float(start),
        end_time=float(start) + 1.0,
        anchor_bbox=BBox(40, 520, 280, 590, 320, 640),
        source_text=text,
    )


def test_subtitle_processor_skips_vietnamese_and_counters():
    translator = _Translator({"Hello world": "Hello world", "你好": "Xin chào"})
    events = [
        _caption("vi", "Kì lạ thật", 0),
        _caption("noise", "122", 20),
        _caption("zh", "你好", 40),
        _caption("echo", "Hello world", 60),
    ]
    processor = SubtitleProcessor(
        segmenter=_Segmenter(events),
        translation_service=translator,
    )

    result = {event.event_id: event for event in processor.process_tracks([object()])}

    assert translator.seen == ["你好", "Hello world"]
    assert result["vi"].replacement_status == "SOURCE_IS_TARGET"
    assert result["vi"].blur_applied is False
    assert result["vi"].translated_text == ""
    assert result["noise"].replacement_status == "OCR_EMPTY"
    assert result["zh"].replacement_status == "READY"
    assert result["zh"].translated_text == "Xin chào"
    assert result["echo"].replacement_status == "TRANSLATION_UNCHANGED"
    assert result["echo"].overlay_applied is False


def _segment(segment_id, text, start):
    return SpeechSegment(
        segment_id=segment_id,
        start_time=start,
        end_time=start + 1.0,
        source_text=text,
    )


def test_dub_keeps_vietnamese_and_drops_untranslated_source(tmp_path):
    translator = _Translator({"你好": "你好", "再见": "Tạm biệt"})
    pipeline = AudioPipeline(translation_service=translator)
    request = AudioRequest(
        enabled=True,
        audio_path="speech.wav",
        target_language="vi",
        dubbing_enabled=False,
        segments=[
            _segment("vi", "Kì lạ thật", 0.0),
            _segment("echo", "你好", 2.0),
            _segment("zh", "再见", 4.0),
        ],
    )

    result = pipeline.process(request, str(tmp_path))
    spoken = {seg.segment_id: seg.spoken_text for seg in result.segments}

    assert translator.seen == ["你好", "再见"]
    assert spoken["vi"] == "Kì lạ thật"
    assert spoken["echo"] == ""
    assert spoken["zh"] == "Tạm biệt"
    assert any("echo" in warning for warning in result.warnings)


class _CaptureTTS:
    def __init__(self):
        self.lines = []

    def synthesize(self, text, output_path, voice):
        self.lines.append(text)
        raise RuntimeError("captured")


def test_dub_does_not_speak_source_language(tmp_path):
    translator = _Translator({"你好": "Xin chào"})
    tts = _CaptureTTS()
    pipeline = AudioPipeline(translation_service=translator, tts_provider=tts)
    request = AudioRequest(
        enabled=True,
        audio_path="speech.wav",
        target_language="vi",
        dubbing_enabled=True,
        segments=[
            _segment("vi", "Kì lạ thật", 0.0),
            _segment("zh", "你好", 2.0),
        ],
    )

    result = pipeline.process(request, str(tmp_path))

    assert tts.lines == ["Kì lạ thật", "Xin chào"]
    assert result.status == "PARTIAL"
