import pytest

from src.domain.entities.frame import Frame
from src.domain.entities.segment import Segment
from src.infrastructure.gemini.exceptions import GeminiRateLimitError, GeminiValidationError
from src.infrastructure.providers.provider_fallback import (
    FallbackOCRProvider,
    FallbackSpeechProvider,
)


class _PrimaryOCR:
    validation = [{"attempt": 1}]

    def detect_text(self, frames):
        raise GeminiRateLimitError("quota exhausted")


class _PrimarySpeech:
    validation = [{"attempt": 1}]

    def transcribe(self, audio_path, source_language):
        raise GeminiRateLimitError("quota exhausted")


class _FallbackOCR:
    def __init__(self):
        self.called = False

    def detect_text(self, frames):
        self.called = True
        return [Segment("ocr", 0, 1, "本地", source="ocr")]


class _FallbackSpeech:
    def __init__(self):
        self.called = False

    def transcribe(self, audio_path, source_language):
        self.called = True
        return [Segment("speech", 0, 1, "local", source="speech")]


def test_ocr_rate_limit_uses_paddle_fallback_without_merging():
    fallback = _FallbackOCR()
    provider = FallbackOCRProvider(_PrimaryOCR(), fallback, "gemini_vision", "paddle")

    result = provider.detect_text([Frame(timestamp=0, image_path="unused")])

    assert [segment.text for segment in result] == ["本地"]
    assert fallback.called is True
    assert provider.provenance == {
        "configured_provider": "gemini_vision",
        "effective_provider": "paddle",
        "fallback_used": True,
        "fallback_trigger": "quota_exhausted",
        "primary_error_type": "GeminiRateLimitError",
        "primary_retry_count": 1,
        "fallback_success": True,
    }


def test_asr_rate_limit_uses_whisper_fallback_without_merging():
    fallback = _FallbackSpeech()
    provider = FallbackSpeechProvider(
        _PrimarySpeech(), fallback, "gemini_audio", "faster-whisper"
    )

    result = provider.transcribe("unused.wav", "zh")

    assert [segment.text for segment in result] == ["local"]
    assert fallback.called is True
    assert provider.provenance["effective_provider"] == "faster-whisper"


def test_invalid_media_does_not_use_fallback():
    class InvalidPrimary:
        validation = []

        def transcribe(self, audio_path, source_language):
            raise GeminiValidationError("Invalid or corrupt WAV file")

    fallback = _FallbackSpeech()
    provider = FallbackSpeechProvider(
        InvalidPrimary(), fallback, "gemini_audio", "faster-whisper"
    )

    with pytest.raises(GeminiValidationError):
        provider.transcribe("bad.wav", "zh")
    assert fallback.called is False


def test_phase_f_defaults_and_shared_runtime():
    from src.infrastructure.factory.provider_factory import ProviderFactory
    from src.infrastructure.providers.ocr.gemini_vision_ocr_provider import GeminiVisionOCRProvider
    from src.infrastructure.providers.speech.gemini_audio_speech_provider import GeminiAudioSpeechProvider
    from src.infrastructure.providers.translation.gemini_translation_provider import GeminiTranslationProvider

    factory = ProviderFactory(
        config={
            "defaults": {"video": "mock", "translation": "mock"},
        },
        gemini_api_key="test-key"
    )
    assert factory.default_name("ocr") == "gemini_vision"
    assert factory.default_name("speech") == "gemini_audio"

    ocr_p = factory.create_ocr_provider()
    speech_p = factory.create_speech_provider()
    trans_p = factory.create_translation_provider(name="gemini")

    actual_ocr = ocr_p.primary if isinstance(ocr_p, FallbackOCRProvider) else ocr_p
    actual_speech = speech_p.primary if isinstance(speech_p, FallbackSpeechProvider) else speech_p

    assert isinstance(actual_ocr, GeminiVisionOCRProvider)
    assert isinstance(actual_speech, GeminiAudioSpeechProvider)
    assert isinstance(trans_p, GeminiTranslationProvider)

    assert actual_ocr.runtime is actual_speech.runtime
    assert actual_ocr.runtime is trans_p._runtime


def test_phase_f_api_key_fail_fast_and_mock_bypass(monkeypatch):
    from src.interfaces.composition.application_builder import ApplicationBuilder
    from src.infrastructure.gemini.exceptions import ProviderConfigurationError
    from src.interfaces.config.config_loader import ConfigLoader

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    original_load = ConfigLoader._load_yaml
    monkeypatch.setattr(
        ConfigLoader,
        "_load_yaml",
        lambda *args: {} if "config.yml" in str(args[-1]) else original_load(args[-1]),
    )
    builder = ApplicationBuilder(
        config_path="config/providers.example.yaml",
        speech_provider=None,
        ocr_provider=None,
    )
    with pytest.raises(ProviderConfigurationError, match="GEMINI_API_KEY is missing"):
        builder.build()

    builder_mock = ApplicationBuilder(
        video_provider="mock",
        speech_provider="mock",
        translation_provider="mock",
        subtitle_provider="mock",
        ocr_provider="mock",
        frame_provider="mock",
    )
    app = builder_mock.build()


def test_phase_f_explicit_overrides():
    from src.infrastructure.factory.provider_factory import ProviderFactory

    factory = ProviderFactory(
        config={
            "defaults": {"ocr": "gemini_vision", "speech": "gemini_audio"},
        },
        gemini_api_key="test-key"
    )
    ocr_p = factory.create_ocr_provider(name="paddle")
    speech_p = factory.create_speech_provider(name="faster-whisper")

    assert not isinstance(ocr_p, FallbackOCRProvider)
    assert ocr_p.__class__.__name__ == "PaddleOCRProvider"
    assert not isinstance(speech_p, FallbackSpeechProvider)
    assert speech_p.__class__.__name__ == "FasterWhisperSpeechProvider"
