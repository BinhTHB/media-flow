import pytest
from src.domain.entities.segment import Segment
from src.application.use_cases.translate_segments import TranslateSegmentsUseCase
from src.application.use_cases.generate_subtitle import GenerateSubtitleUseCase
from src.application.use_cases.detect_onscreen_text import DetectOnscreenTextUseCase
from src.application.use_cases.extract_audio import ExtractAudioUseCase
from src.application.use_cases.extract_frames import ExtractFramesUseCase
from src.application.use_cases.render_video import RenderVideoUseCase
from src.application.use_cases.transcribe_speech import TranscribeSpeechUseCase
from src.domain.entities.frame import Frame
from src.infrastructure.providers.ocr.mock_ocr_provider import MockOCRProvider
from src.infrastructure.providers.translation.mock_translation_provider import MockTranslationProvider
from src.infrastructure.providers.subtitle.mock_subtitle_provider import MockSubtitleProvider
from src.infrastructure.providers.speech.mock_speech_provider import MockSpeechProvider
from src.infrastructure.providers.video.mock_frame_extraction_provider import (
    MockFrameExtractionProvider,
)
from src.infrastructure.providers.video.mock_video_provider import MockVideoProvider

def test_translate_segments_use_case(mock_segments):
    provider = MockTranslationProvider()
    uc = TranslateSegmentsUseCase(provider)
    translated = uc.execute(mock_segments, "vi")
    assert len(translated) == 2
    # Note: mock translation provider uses fixed mapping, so may not translate "Hello" -> "Chào..."
    # but it will add translated_text with fallback
    assert translated[0].translated_text != ""
    assert translated[1].translated_text != ""

def test_generate_subtitle_use_case(mock_segments):
    provider = MockSubtitleProvider()
    uc = GenerateSubtitleUseCase(provider)
    # Ensure they have translated texts
    mock_segments[0].translated_text = "Chào mọi người"
    mock_segments[1].translated_text = "Chào thế giới"
    srt_content = uc.execute(mock_segments)
    assert "00:00:00,000 --> 00:00:01,000" in srt_content
    assert "Chào mọi người" in srt_content
    assert "Chào thế giới" in srt_content


def test_extract_audio_use_case_delegates_to_video_provider(tmp_path):
    audio_path = tmp_path / "audio.wav"

    result = ExtractAudioUseCase(MockVideoProvider()).execute(str(audio_path))

    assert result == str(tmp_path / "audio_extracted.wav")
    assert (tmp_path / "audio_extracted.wav").is_file()


def test_transcribe_speech_use_case_delegates_to_speech_provider():
    segments = TranscribeSpeechUseCase(MockSpeechProvider()).execute("audio.wav", "zh")

    assert [segment.id for segment in segments] == ["speech_1", "speech_2"]
    assert all(segment.source == "speech" for segment in segments)


def test_extract_frames_use_case_delegates_to_frame_provider(tmp_path):
    frames = ExtractFramesUseCase(MockFrameExtractionProvider()).execute(
        "video.mp4", 1.0, str(tmp_path)
    )

    assert len(frames) == 2
    assert all((tmp_path / f"frame_{index:04d}.jpg").is_file() for index in (0, 1))


def test_detect_onscreen_text_use_case_delegates_to_ocr_provider(tmp_path):
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_text("mock", encoding="utf-8")
    frames = [Frame(timestamp=1.0, image_path=str(frame_path))]

    segments = DetectOnscreenTextUseCase(MockOCRProvider()).execute(frames)

    assert len(segments) == 1
    assert segments[0].source == "ocr"
    assert segments[0].start == 1.0


def test_render_video_use_case_delegates_to_video_provider(tmp_path):
    output_path = tmp_path / "rendered.mp4"

    result = RenderVideoUseCase(MockVideoProvider()).execute(
        video_path="video.mp4",
        subtitle_path="subtitle.srt",
        output_path=str(output_path),
    )

    assert result == str(output_path)
    assert output_path.read_text(encoding="utf-8") == "mock rendered video"
