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
    is_collapsed_speech,
    repair_asr_phrases,
    restore_introduced_names,
    speech_defect_tokens,
    is_dangling_fragment,
    is_effectively_identical,
    is_production_credit,
    should_skip_spoken_translation,
)
from src.features.visual.pipeline import VisualPipeline
from src.features.visual.subtitle.processor import SubtitleProcessor


_COLLAPSED_ASR = (
    "b n Ngay gi c v h l b d k nh t b tr d t t r mu tr x l t n qu "
    "Mu r s n gi cho c kh th M tu sau t tr cao nh xu c c th m t th"
)


def test_collapsed_asr_is_not_spoken_or_captioned():
    assert is_collapsed_speech(_COLLAPSED_ASR)
    assert not is_collapsed_speech(
        "biến cái khô cằn nóng bức thành lợi thế. Vụ thứ hai, ông không bán cà chua nữa,"
    )
    assert accept_spoken_line(
        _COLLAPSED_ASR, _COLLAPSED_ASR, "vi", detected_language="Vietnamese"
    ) == ""
    events = VisualPipeline()._captions_from_speech(
        [],
        [SpeechSegment("S00014", 64.8, 81.07, _COLLAPSED_ASR, "Vietnamese")],
        "vi",
        width=720,
        height=720,
        fps=24,
    )
    assert events[0].overlay_applied is False
    assert events[0].translated_text == ""


def test_voweled_asr_misses_are_defects_and_real_words_are_not():
    assert "tisa" in [token.casefold() for token in speech_defect_tokens("bánh tisa ở Ý")]
    assert speech_defect_tokens("bững vàng với những gì mình đang làm.")
    assert "re" in [token.casefold() for token in speech_defect_tokens("hàng xóm từng re biểu")]
    assert speech_defect_tokens("cà chua rẻ như cho, những người cũng đứng.") == []
    assert speech_defect_tokens("Ông tên Mehmet, một nông dân.") == ["Mehmet"]
    assert speech_defect_tokens("trong chiếc burger ở Mỹ.") == []
    assert repair_asr_phrases("Dù người đời có rẻ biểu đến đâu.") == "Dù người đời có dè bỉu đến đâu."
    assert repair_asr_phrases("hàng xóm từng rèm bỉu") == "hàng xóm từng dè bỉu"
    assert repair_asr_phrases("cà chua rẻ như cho") == "cà chua rẻ như cho"
    assert repair_asr_phrases("Ông bỏ ngoài tay, ngày này qua ngày nọ.") == (
        "Ông bỏ ngoài tai, ngày này qua ngày nọ."
    )
    assert repair_asr_phrases("hàng xóm từng re biểu, ông chia sẻ") == (
        "hàng xóm từng dè bỉu, ông chia sẻ"
    )
    assert repair_asr_phrases("bánh tisa ở Ý, trong lọ dầu") == "bánh pizza ở Ý, trong lọ dầu"
    assert speech_defect_tokens("nếm một miếng là đặt luôn cả xe tải.") == []
    assert repair_asr_phrases("bững vàng với những gì mình đang làm.") == (
        "vững vàng với những gì mình đang làm."
    )
    assert repair_asr_phrases("những người cũng đứng.") == "những người cũng đứng."
    assert restore_introduced_names(
        ["Ông tên Mehmet, một nông dân.", "nhưng mê mét không bỏ"]
    ) == ["Ông tên Mehmet, một nông dân.", "nhưng Mehmet không bỏ"]
    assert accept_spoken_line(
        "bánh tisa ở Ý", "bánh tisa ở Ý", "vi", detected_language="Vietnamese"
    ) == "bánh tisa ở Ý"


def test_caption_and_language_gates():
    assert is_already_target_language("Kì lạ thật", "vi")
    assert not is_already_target_language("Hello there", "vi")
    assert not is_already_target_language("你好", "vi")
    assert not is_already_target_language("tranh nhau.", "vi")
    assert should_skip_spoken_translation(
        "tranh nhau.", "vi", detected_language="Vietnamese"
    )
    assert not should_skip_spoken_translation(
        "你好", "vi", detected_language="Vietnamese"
    )
    assert not should_skip_spoken_translation(
        "Hello there", "vi", detected_language="auto"
    )
    assert not is_caption_worthy("122")
    assert not is_caption_worthy("ab 1234")
    assert is_caption_worthy("你好")
    assert is_effectively_identical("Kỳ lạ!", "ky la")
    assert accept_spoken_line("你好", "你好", "vi") == ""
    assert accept_spoken_line("你好", "Xin chào", "vi") == "Xin chào"
    assert accept_spoken_line("Kì lạ thật", "", "vi") == "Kì lạ thật"
    assert accept_spoken_line("Kì lạ thật", "Lạ thật", "vi") == "Lạ thật"
    assert accept_spoken_line("tranh nhau.", "tranh nhau.", "vi") == ""
    assert (
        accept_spoken_line(
            "tranh nhau.",
            "tranh nhau.",
            "vi",
            detected_language="Vietnamese",
        )
        == "tranh nhau."
    )
    assert is_production_credit("※人形です")
    assert is_production_credit("西瓜君AI制作")
    assert not is_production_credit("海军总部大将")
    assert not is_production_credit("Nên phải có người")
    assert is_dangling_fragment("那是", box_width=100, frame_width=1920, duration=0.9)
    assert not is_dangling_fragment("赤犬", box_width=169, frame_width=1920, duration=1.47)
    assert not is_dangling_fragment("青雉", box_width=176, frame_width=1920, duration=2.87)


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
        _caption("credit", "※人形です", 80),
    ]
    processor = SubtitleProcessor(
        segmenter=_Segmenter(events),
        translation_service=translator,
    )

    result = {event.event_id: event for event in processor.process_tracks([object()])}

    assert translator.seen == ["你好", "Hello world"]
    assert result["vi"].replacement_status == "READY"
    assert result["vi"].blur_applied is True
    assert result["vi"].overlay_applied is True
    assert result["vi"].translated_text == "Kì lạ thật"
    assert result["noise"].replacement_status == "OCR_EMPTY"
    assert result["zh"].replacement_status == "READY"
    assert result["zh"].translated_text == "Xin chào"
    assert result["echo"].replacement_status == "TRANSLATION_UNCHANGED"
    assert result["echo"].overlay_applied is False
    assert result["credit"].replacement_status == "CREDIT"
    assert result["credit"].translated_text == ""
    assert result["credit"].blur_applied is True
    assert result["credit"].overlay_applied is False


def _segment(segment_id, text, start):
    return SpeechSegment(
        segment_id=segment_id,
        start_time=start,
        end_time=start + 1.0,
        source_text=text,
    )


def test_vietnamese_source_with_a_covered_credit_is_a_successful_visual_pass():
    from types import SimpleNamespace

    from src.features.visual.pipeline import replacement_outcome

    kept = SimpleNamespace(replacement_status="SOURCE_IS_TARGET", overlay_applied=False)
    credit = SimpleNamespace(replacement_status="CREDIT", overlay_applied=False)
    status, warning = replacement_outcome([kept, credit])
    assert status == "SUCCESS"
    assert warning is None

    missed = SimpleNamespace(replacement_status="TRANSLATION_FAILED", overlay_applied=False)
    failed, detail = replacement_outcome([missed])
    assert failed == "FAILED"
    assert detail


def test_dangling_fragment_is_covered_without_a_caption():
    event = _caption("chip", "那是", 15.07)
    event.anchor_bbox = BBox(907, 991, 1007, 1042, 1920, 1080)
    event.end_time = 15.97
    translator = _Translator({"那是": "Đó là"})
    processor = SubtitleProcessor(
        segmenter=_Segmenter([event]),
        translation_service=translator,
    )
    result = processor.process_tracks([object()])[0]
    assert result.replacement_status == "FRAGMENT"
    assert result.translated_text == ""
    assert result.blur_applied is True
    assert result.overlay_applied is False


def test_asr_vietnamese_without_diacritics_does_not_call_translator(tmp_path):
    translator = _Translator({"tranh nhau.": "không được gọi"})
    pipeline = AudioPipeline(translation_service=translator)
    request = AudioRequest(
        enabled=True,
        audio_path="speech.wav",
        source_language="auto",
        target_language="vi",
        dubbing_enabled=False,
        segments=[
            SpeechSegment(
                segment_id="marked",
                start_time=0.0,
                end_time=3.0,
                source_text="Những gốc cây cảnh tiên tỷ đứng giữa triển lám bonsai thế giới",
                source_language="Vietnamese",
            ),
            SpeechSegment(
                segment_id="plain",
                start_time=3.0,
                end_time=4.0,
                source_text="tranh nhau.",
                source_language="Vietnamese",
            ),
        ],
    )

    result = pipeline.process(request, str(tmp_path))
    spoken = {seg.segment_id: seg.spoken_text for seg in result.segments}

    assert translator.seen == []
    assert spoken["marked"].startswith("Những gốc cây")
    assert spoken["plain"] == "tranh nhau."
    assert result.warnings == []


def test_empty_translation_batch_does_not_abort_the_dub(tmp_path):
    class _Boom:
        def translate_texts(self, texts, target_language="vi", source_language="auto"):
            raise RuntimeError(
                "All translation providers failed. gemini: provider returned an empty translation batch"
            )

    pipeline = AudioPipeline(translation_service=_Boom())
    request = AudioRequest(
        enabled=True,
        audio_path="speech.wav",
        target_language="vi",
        dubbing_enabled=False,
        segments=[
            _segment("vi", "Kì lạ thật", 0.0),
            _segment("zh", "你好", 2.0),
        ],
    )

    result = pipeline.process(request, str(tmp_path))
    spoken = {seg.segment_id: seg.spoken_text for seg in result.segments}

    assert spoken["vi"] == "Kì lạ thật"
    assert spoken["zh"] == ""
    assert any("empty translation batch" in warning for warning in result.warnings)


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


def test_short_vietnamese_tail_is_folded_into_previous_tts(tmp_path):
    tts = _CaptureTTS()
    pipeline = AudioPipeline(tts_provider=tts)
    request = AudioRequest(
        enabled=True,
        audio_path="speech.wav",
        source_language="vi",
        target_language="vi",
        dubbing_enabled=True,
        segments=[
            SpeechSegment(
                segment_id="body",
                start_time=11.86,
                end_time=14.54,
                source_text="Mỗi năm, hàng ngàn người chơi cây khắp thế giới",
                source_language="Vietnamese",
            ),
            SpeechSegment(
                segment_id="tail",
                start_time=14.54,
                end_time=14.96,
                source_text="tranh nhau.",
                source_language="Vietnamese",
            ),
        ],
    )

    pipeline.process(request, str(tmp_path))

    assert set(tts.lines) == {
        "Mỗi năm, hàng ngàn người chơi cây khắp thế giới tranh nhau."
    }


def test_a_failed_tts_line_is_spoken_on_the_second_try(tmp_path):
    from pathlib import Path

    class _FlakyTTS:
        def __init__(self):
            self.calls = 0

        def synthesize(self, text, output_path, voice):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("No audio was received")
            Path(output_path).write_bytes(b"ID3audio")
            return output_path

    class _Composer:
        max_speedup = 1.8

        def trim_tts_silence(self, source, output):
            Path(output).write_bytes(b"RIFFaudio")
            return output

        def probe_duration(self, path):
            return 1.0

        def compose_dubbed_timeline(self, tts_clips, total_duration, output_audio_path, **_kwargs):
            self.clips = list(tts_clips)
            return output_audio_path

    tts = _FlakyTTS()
    composer = _Composer()
    pipeline = AudioPipeline(tts_provider=tts, composer=composer)
    request = AudioRequest(
        enabled=True,
        audio_path="speech.wav",
        source_language="vi",
        target_language="vi",
        dubbing_enabled=True,
        segments=[
            SpeechSegment(
                segment_id="line",
                start_time=0.0,
                end_time=2.0,
                source_text="vững vàng với những gì mình đang làm.",
                source_language="Vietnamese",
            )
        ],
    )

    result = pipeline.process(request, str(tmp_path))

    assert tts.calls == 2
    assert result.status == "SUCCESS"
    assert [clip.segment_id for clip in composer.clips] == ["line"]
    assert not any("TTS synthesis failed" in warning for warning in result.warnings)


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

    assert set(tts.lines) == {"Kì lạ thật", "Xin chào"}
    assert result.status == "PARTIAL"
