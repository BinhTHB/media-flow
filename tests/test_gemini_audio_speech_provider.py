import json
import wave
import pytest
from pathlib import Path
import shutil

from src.application.ports.speech_provider import SpeechProvider
from src.domain.entities.segment import Segment
from src.infrastructure.factory.provider_factory import ProviderFactory
from src.infrastructure.gemini.runtime import GeminiRuntime, RuntimeConfig
from src.infrastructure.providers.speech.gemini_audio_speech_provider import GeminiAudioSpeechProvider
from src.infrastructure.providers.ocr.gemini_vision_ocr_provider import GeminiVisionOCRProvider
from src.infrastructure.providers.translation.gemini_translation_provider import GeminiTranslationProvider
from src.shared.exceptions import ProviderError
from src.infrastructure.gemini.exceptions import GeminiValidationError

class FakeTransport:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def send(self, method, url, *, headers, data=None, timeout=60.0):
        self.calls.append((method, json.loads(data.decode("utf-8")) if data else None))
        if method == "GET":
            return {"name": "models/test", "supportedGenerationMethods": ["generateContent"], "modelStage": "STABLE"}
        if not self.payloads:
            raise AssertionError("Unexpected REST request")

        payload = self.payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return {
            "candidates": [{"content": {"parts": [{"text": json.dumps(payload, ensure_ascii=False)}]}}],
            "modelStatus": {"modelStage": "STABLE"},
        }

def create_dummy_wav(path, duration_seconds=1.0, frame_rate=16000):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(frame_rate)
        num_frames = int(duration_seconds * frame_rate)
        wav.writeframes(b"\x00" * (num_frames * 2))

def _provider(tmp_path, payloads, **kwargs):
    runtime = GeminiRuntime.create(RuntimeConfig(
        api_key="fake-api-key",
        cache_dir=str(tmp_path / "cache"),
        cache_enabled=True
    ))
    provider = GeminiAudioSpeechProvider(runtime=runtime, api_key="fake-api-key", **kwargs)
    transport = FakeTransport(payloads)
    runtime.get_client(provider.model, role="primary", api_key="fake-api-key")._transport = transport
    runtime.get_client(provider.escalation_model, role="escalation", api_key="fake-api-key")._transport = transport
    return provider, transport

def test_implements_speech_provider_and_shared_runtime(tmp_path):
    runtime = GeminiRuntime.create()
    provider = GeminiAudioSpeechProvider(runtime=runtime)
    assert isinstance(provider, SpeechProvider)
    assert provider.runtime is runtime

def test_duration_derives_from_wave_header(tmp_path):
    path = tmp_path / "test.wav"
    create_dummy_wav(path, duration_seconds=2.5, frame_rate=8000)
    provider = GeminiAudioSpeechProvider()
    metadata = provider._inspect_wav(str(path))
    assert metadata["duration"] == 2.5
    assert metadata["frame_count"] == 20000
    assert metadata["frame_rate"] == 8000

def test_corrupt_wav_rejected(tmp_path):
    path = tmp_path / "corrupt.wav"
    path.write_bytes(b"invalid data")
    provider = GeminiAudioSpeechProvider()
    with pytest.raises(GeminiValidationError):
        provider._inspect_wav(str(path))

def test_projected_base64_size_helper():
    provider = GeminiAudioSpeechProvider(inline_payload_safety_limit_bytes=100)
    assert provider._exceeds_inline_limit(b"x" * 100) is True
    assert provider._exceeds_inline_limit(b"x" * 5) is False

def test_actual_json_body_payload_guard(tmp_path):
    path = tmp_path / "test.wav"
    create_dummy_wav(path, duration_seconds=0.1)
    provider, transport = _provider(tmp_path, [], inline_payload_safety_limit_bytes=10)
    with pytest.raises(ProviderError, match="WAV payload size exceeds inline limit"):
        provider.transcribe(str(path), "en")

def test_inline_base64_request(tmp_path):
    path = tmp_path / "test.wav"
    create_dummy_wav(path, duration_seconds=0.5)
    payload = {
        "source_language": "en",
        "segments": [{"start": 0.0, "end": 0.5, "speaker": "SPEAKER_1", "text_verbatim": "test", "speech_status": "clear"}]
    }
    provider, transport = _provider(tmp_path, [payload])
    segments = provider.transcribe(str(path), "en")
    assert len(segments) == 1
    assert segments[0].text == "test"
    req_body = transport.calls[0][1]
    inline_data = req_body["contents"][0]["parts"][2]["inline_data"]
    assert inline_data["mime_type"] == "audio/wav"
    assert len(inline_data["data"]) > 0

def test_valid_unicode_cjk_verbatim(tmp_path):
    path = tmp_path / "test.wav"
    create_dummy_wav(path, duration_seconds=1.0)
    payload = {
        "source_language": "zh",
        "segments": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_1", "text_verbatim": "英国/因果", "speech_status": "clear"}]
    }
    provider, transport = _provider(tmp_path, [payload])
    segments = provider.transcribe(str(path), "zh")
    assert len(segments) == 1
    assert segments[0].text == "英国/因果"
    assert "英国" in provider.build_prompt()

def test_empty_transcript(tmp_path):
    path = tmp_path / "test.wav"
    create_dummy_wav(path, duration_seconds=1.0)
    payload = {
        "source_language": "en",
        "segments": []
    }
    provider, transport = _provider(tmp_path, [payload])
    segments = provider.transcribe(str(path), "en")
    assert len(segments) == 0

def test_invalid_timestamps_rejected(tmp_path):
    path = tmp_path / "test.wav"
    create_dummy_wav(path, duration_seconds=1.0)

    payload_decreasing = {
        "source_language": "en",
        "segments": [
            {"start": 0.5, "end": 0.3, "speaker": "SPEAKER_1", "text_verbatim": "decreasing", "speech_status": "clear"}
        ]
    }
    provider, transport = _provider(tmp_path, [payload_decreasing, payload_decreasing])
    with pytest.raises(ProviderError):
        provider.transcribe(str(path), "en")

    payload_nonmonotonic = {
        "source_language": "en",
        "segments": [
            {"start": 0.5, "end": 0.8, "speaker": "SPEAKER_1", "text_verbatim": "first", "speech_status": "clear"},
            {"start": 0.4, "end": 0.9, "speaker": "SPEAKER_1", "text_verbatim": "second", "speech_status": "clear"}
        ]
    }
    provider, transport = _provider(tmp_path, [payload_nonmonotonic, payload_nonmonotonic])
    with pytest.raises(ProviderError):
        provider.transcribe(str(path), "en")

def test_cache_hit_avoids_second_rest(tmp_path):
    path = tmp_path / "test.wav"
    create_dummy_wav(path, duration_seconds=1.0)
    payload = {
        "source_language": "en",
        "segments": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_1", "text_verbatim": "cached", "speech_status": "clear"}]
    }
    provider, transport = _provider(tmp_path, [payload])

    res1 = provider.transcribe(str(path), "en")
    assert len(transport.calls) == 1

    res2 = provider.transcribe(str(path), "en")
    assert len(transport.calls) == 1
    assert res1[0].text == res2[0].text

def test_chunk_fallback(tmp_path):
    path = tmp_path / "long.wav"
    create_dummy_wav(path, duration_seconds=5.0, frame_rate=8000)

    p0 = {
        "source_language": "en",
        "segments": [{"start": 0.2, "end": 1.0, "speaker": "SPEAKER_1", "text_verbatim": "chunk zero", "speech_status": "clear"}]
    }
    p1 = {
        "source_language": "en",
        "segments": [{"start": 0.5, "end": 1.5, "speaker": "SPEAKER_1", "text_verbatim": "chunk one", "speech_status": "clear"}]
    }
    p2 = {
        "source_language": "en",
        "segments": [{"start": 0.5, "end": 1.5, "speaker": "SPEAKER_1", "text_verbatim": "chunk two", "speech_status": "clear"}]
    }
    provider, transport = _provider(tmp_path, [p0, p1, p2],
                                    full_audio_max_seconds=3.0,
                                    chunk_fallback_enabled=True,
                                    chunk_duration_seconds=2.0,
                                    chunk_overlap_seconds=0.5)

    segments = provider.transcribe(str(path), "en")
    assert len(segments) == 3
    assert segments[0].start == pytest.approx(0.2)
    assert segments[1].start == pytest.approx(2.0)
    assert segments[2].start == pytest.approx(3.5)

    chunk_dir = tmp_path / "qualification" / "gemini_asr" / "tmp_chunks"
    assert not chunk_dir.exists()

def test_chunk_fallback_cleans_temp_chunks_on_exception(tmp_path):
    path = tmp_path / "long_fail.wav"
    create_dummy_wav(path, duration_seconds=5.0, frame_rate=8000)

    p0 = {
        "source_language": "en",
        "segments": [{"start": 0.2, "end": 1.0, "speaker": "SPEAKER_1", "text_verbatim": "chunk zero", "speech_status": "clear"}]
    }
    err = ProviderError("REST API fail")
    provider, transport = _provider(tmp_path, [p0, err],
                                    full_audio_max_seconds=3.0,
                                    chunk_fallback_enabled=True,
                                    chunk_duration_seconds=2.0,
                                    chunk_overlap_seconds=0.5,
                                    escalation_enabled=False)

    with pytest.raises(ProviderError):
        provider.transcribe(str(path), "en")

    chunk_dir = tmp_path / "qualification" / "gemini_asr" / "tmp_chunks"
    assert not chunk_dir.exists()

def test_file_manager_capabilities():
    provider = GeminiAudioSpeechProvider()
    assert provider.runtime.file_manager.supports_inline is True
    assert provider.runtime.file_manager.supports_remote_upload is False

def test_registry_shares_runtime_instance():
    config = {
        "defaults": {
            "speech": "gemini_audio",
            "ocr": "gemini_vision",
            "translation": "gemini"
        },
        "providers": {
            "speech": {"gemini_audio": {}},
            "ocr": {"gemini_vision": {}},
            "translation": {"gemini": {}}
        }
    }
    factory = ProviderFactory(config=config, gemini_api_key="shared-key")
    speech_prov = factory.create_speech_provider()
    ocr_prov = factory.create_ocr_provider()
    trans_prov = factory.create_translation_provider()

    assert isinstance(speech_prov, GeminiAudioSpeechProvider)
    assert isinstance(ocr_prov, GeminiVisionOCRProvider)
    assert isinstance(trans_prov, GeminiTranslationProvider)

    assert speech_prov.runtime is ocr_prov.runtime
    assert speech_prov.runtime is trans_prov._runtime

def test_write_qualification_artifacts(tmp_path):
    provider = GeminiAudioSpeechProvider()
    segments = [
        Segment(id="speech_0_0", start=0.0, end=1.0, text="test", source="speech")
    ]
    provider.write_qualification_artifacts("job_test", segments, output_dir=str(tmp_path))

    artifact_dir = tmp_path / "job_test" / "qualification" / "gemini_asr"
    assert artifact_dir.exists()
    assert (artifact_dir / "transcript.json").exists()
    assert (artifact_dir / "provenance.json").exists()
    assert (artifact_dir / "usage.json").exists()
    assert (artifact_dir / "validation.json").exists()
    assert (artifact_dir / "comparison.json").exists()
