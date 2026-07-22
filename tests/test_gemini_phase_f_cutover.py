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


def test_raw_gemini_provider_provenance_is_serializable():
    from src.application.orchestrator.job_orchestrator import JobOrchestrator

    class RawGeminiProvider:
        provenance = [{"model": "gemini-3.5-flash-lite", "cache_hit": False}]
        validation = []
        model = "gemini-3.5-flash-lite"
        _escalation_requests = 0

    result = JobOrchestrator._provider_provenance(RawGeminiProvider())

    assert result["exact_model_id"] == "gemini-3.5-flash-lite"
    assert result["request_count"] == 1
    assert result["fallback_used"] is False
    assert result["request_provenance"] == RawGeminiProvider.provenance


@pytest.mark.parametrize("provenance, expected", [
    ({"fallback_used": True}, {"fallback_used": True}),
    ([], {"request_provenance": []}),
    (None, {}),
])
def test_provider_provenance_accepts_mapping_empty_list_and_missing(provenance, expected):
    from src.application.orchestrator.job_orchestrator import JobOrchestrator

    class Provider:
        pass

    provider = Provider()
    if provenance is not None:
        provider.provenance = provenance

    result = JobOrchestrator._provider_provenance(provider)

    for key, value in expected.items():
        assert result[key] == value


def test_gemini_ocr_without_confidence_bypasses_confidence_filter():
    from src.application.use_cases.filter_ocr_segments import FilterOCRSegmentsUseCase

    segment = Segment(
        "ocr-1",
        0,
        1,
        "字幕内容",
        source="ocr",
        confidence=0.0,
        confidence_available=False,
    )

    kept, metrics = FilterOCRSegmentsUseCase().execute([segment], "zh")

    assert kept == [segment]
    assert metrics["dropped_by_confidence"] == 0


def test_ocr_bbox_becomes_display_region_with_fallback_confidence():
    from src.application.orchestrator.job_orchestrator import JobOrchestrator

    segment = Segment(
        "ocr-1",
        1,
        2,
        "字幕内容",
        source="ocr",
        confidence=0.0,
        confidence_available=False,
        bbox={"left": 100, "top": 200, "right": 500, "bottom": 300},
    )

    regions = JobOrchestrator._ocr_regions([segment], 1000, 500)

    assert len(regions) == 1
    assert regions[0].to_dict() == {
        "start_time": 1,
        "end_time": 2,
        "x": 0.1,
        "y": 0.4,
        "width": 0.4,
        "height": 0.2,
        "confidence": 1.0,
    }


def test_segment_cache_fields_preserve_gemini_geometry():
    from src.application.orchestrator.job_orchestrator import JobOrchestrator

    item = Segment(
        "ocr-1",
        0,
        1,
        "字幕内容",
        source="ocr",
        confidence_available=False,
        bbox={"left": 10, "top": 20, "right": 30, "bottom": 40},
        metadata={"provider": "gemini"},
    ).to_dict()

    restored = Segment(**JobOrchestrator._segment_fields(item))

    assert restored.confidence_available is False
    assert restored.bbox == item["bbox"]
    assert restored.metadata == item["metadata"]
