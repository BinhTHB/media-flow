from __future__ import annotations

from pathlib import Path

import pytest

from src.core.models.speech_segment import SpeechSegment
from src.infrastructure.speech.router import FallbackASRAdapter
from src.infrastructure.speech.openrouter_corrector import OpenRouterASRCorrector
from src.infrastructure.translation.duration_rewriter import OpenRouterDurationRewriter
from src.infrastructure.translation.openrouter_translation import (
    OpenRouterTranslationAdapter,
    configured_openrouter_keys,
)
from src.infrastructure.translation.router import FallbackTranslationAdapter
from src.infrastructure.tts.fish_audio import FishAudioTTSAdapter
from src.infrastructure.tts.openrouter_tts import OpenRouterTTSAdapter
from src.infrastructure.tts.gemini_flash_live import GeminiFlashLiveTTSAdapter
from src.infrastructure.tts.elevenlabs_tts import ElevenLabsTTSAdapter
from src.infrastructure.elevenlabs_dubbing import configured_elevenlabs_keys
from src.infrastructure.tts.router import FallbackTTSAdapter


def test_openrouter_translation_rotates_key_after_failure(monkeypatch):
    adapter = OpenRouterTranslationAdapter(
        api_keys=["key-a", "key-b"],
        models=["model-a"],
        purpose="subtitle",
    )
    calls = []

    def fake_post(payload, key):
        calls.append((payload["model"], key))
        if key == "key-a":
            raise RuntimeError("HTTP 429")
        return '{"translations":[{"id":0,"text":"Xin chào"}]}'

    monkeypatch.setattr(adapter, "_post", fake_post)
    result = adapter.translate_texts(["你好"], target_language="vi", source_language="zh")
    assert result == ["Xin chào"]
    assert calls == [("model-a", "key-a"), ("model-a", "key-b")]
    assert adapter._cursor == 0  # successful key-b rotates to key-a for the next request


def test_openrouter_translation_falls_back_to_next_model(monkeypatch):
    adapter = OpenRouterTranslationAdapter(
        api_keys=["key-a"],
        models=["model-bad", "model-good"],
        purpose="spoken",
    )
    calls = []

    def fake_post(payload, key):
        calls.append(payload["model"])
        if payload["model"] == "model-bad":
            return "not-json"
        return '{"translations":[{"id":0,"text":"Tạm biệt nhé"}]}'

    monkeypatch.setattr(adapter, "_post", fake_post)
    result = adapter.translate_texts(["拜拜"], target_language="vi", source_language="zh")
    assert result == ["Tạm biệt nhé"]
    assert calls == ["model-bad", "model-good"]


def test_openrouter_spoken_prompt_preserves_short_utterances():
    adapter = OpenRouterTranslationAdapter(api_keys=["x"], models=["m"], purpose="spoken")
    payload = adapter._request_payload(["谢谢", "拜拜"], target_language="vi", source_language="zh")
    prompt = payload["messages"][1]["content"]
    assert "never omit short" in prompt
    assert "Do not merge or split ids" in prompt


def test_openrouter_numbered_env_keys_are_loaded(monkeypatch):
    for name in list(__import__("os").environ):
        if name.startswith("OPENROUTER_API_KEY"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY_1", "key-one")
    monkeypatch.setenv("OPENROUTER_API_KEY_2", "key-two")
    assert configured_openrouter_keys() == ["key-one", "key-two"]


def test_openrouter_fish_maps_ui_voice_to_vietnamese_reference(monkeypatch):
    monkeypatch.setenv("OPENROUTER_FISH_FEMALE_VOICE", "female-ref")
    monkeypatch.setenv("OPENROUTER_FISH_MALE_VOICE", "male-ref")
    assert OpenRouterTTSAdapter._normalize_voice("vi-VN-HoaiMyNeural") == "female-ref"
    assert OpenRouterTTSAdapter._normalize_voice("vi-VN-NamMinhNeural") == "male-ref"
    assert OpenRouterTTSAdapter._normalize_voice("fish:custom-ref") == "custom-ref"


def test_openrouter_fish_style_is_internal_and_configurable(monkeypatch):
    monkeypatch.setenv("OPENROUTER_FISH_STYLE_MODE", "auto")
    styled = OpenRouterTTSAdapter._apply_style("Xin lỗi, tôi giao nhầm đơn rồi.")
    assert styled.startswith("[sincere apologetic conversational tone]")
    assert styled.endswith("Xin lỗi, tôi giao nhầm đơn rồi.")
    monkeypatch.setenv("OPENROUTER_FISH_STYLE_MODE", "off")
    assert OpenRouterTTSAdapter._apply_style("Xin lỗi") == "Xin lỗi"


def test_duration_rewriter_returns_shorter_spoken_copy(monkeypatch):
    adapter = OpenRouterDurationRewriter(api_keys=["k"], models=["m"])
    monkeypatch.setattr(adapter, "_post", lambda payload, key: '{"text":"Tôi giao nhầm đơn rồi."}')
    result = adapter.rewrite_for_duration(
        text="Có một đơn giao hàng tôi giao sai rồi",
        source_text="有个外卖我送错",
        target_duration=1.0,
        current_duration=2.8,
        target_language="vi",
    )
    assert result == "Tôi giao nhầm đơn rồi."


class _FailTranslation:
    def translate_texts(self, texts, target_language="vi", source_language="auto"):
        raise RuntimeError("quota")


class _GoodTranslation:
    def translate_texts(self, texts, target_language="vi", source_language="auto"):
        return [f"VI:{text}" for text in texts]


def test_translation_router_uses_fallback():
    router = FallbackTranslationAdapter([
        ("primary", _FailTranslation()),
        ("fallback", _GoodTranslation()),
    ])
    assert router.translate_texts(["a", "b"]) == ["VI:a", "VI:b"]


def test_translation_router_repairs_cross_segment_leakage_selectively():
    class Primary:
        def translate_texts(self, texts, target_language="vi", source_language="auto"):
            if len(texts) == 1:
                return ["Có một đơn đồ ăn tôi giao nhầm"]
            return ["Xin lỗi", "Xin lỗi Xin lỗi có một đơn hàng tôi giao nhầm"]

    class Fallback:
        def translate_texts(self, texts, target_language="vi", source_language="auto"):
            if len(texts) == 1:
                return ["Có một đơn đồ ăn tôi giao nhầm"]
            return ["Tôi xin lỗi", "Có một đơn đồ ăn tôi giao nhầm"]

    router = FallbackTranslationAdapter([
        ("openrouter", Primary()),
        ("gemini", Fallback()),
    ])
    result = router.translate_texts(["對不起", "有个外卖我送错了"], source_language="zh")
    assert result == ["Xin lỗi", "Có một đơn đồ ăn tôi giao nhầm"]


def test_translation_validator_flags_cross_id_numbers_tokens_and_finance_jargon():
    source = [
        "绞杀国家队",
        "热浪扭曲着空气却还不够让人燥热",
        "因为旁边那台我和别人共用的",
        "Bloomberg终端上",
        "显示的是从2300点",
        "冲到5100多点的股市",
    ]
    translated = [
        'Bóp nghẹt "đội quốc gia"',
        "Sóng nhiệt làm méo không khí nhưng vẫn chưa đủ làm người ta nóng lòng",
        "Bởi vì trên chiếc máy Bloomberg Terminal",
        "Mà tôi dùng chung với người khác bên cạnh",
        "Đang hiển thị thị trường chứng khoán",
        "Tăng vọt từ 2300 điểm lên hơn 5100 điểm",
    ]

    assert FallbackTranslationAdapter._suspicious_indices(source, translated) == [0, 1, 2, 3, 4, 5]


def test_fish_tts_rotates_key_and_writes_audio(monkeypatch, tmp_path):
    adapter = FishAudioTTSAdapter(api_keys=["fish-a", "fish-b"])
    calls = []

    def fake_post(payload, key):
        calls.append(key)
        if key == "fish-a":
            raise RuntimeError("HTTP 429")
        assert payload["format"] == "mp3"
        return b"ID3" + (b"x" * 1024)

    monkeypatch.setattr(adapter, "_post", fake_post)
    output = tmp_path / "voice.mp3"
    result = adapter.synthesize("Xin chào", str(output))
    assert Path(result).read_bytes().startswith(b"ID3")
    assert calls == ["fish-a", "fish-b"]


class _FailTTS:
    def synthesize(self, text, output_path, voice=""):
        raise RuntimeError("capacity")


class _GoodTTS:
    def synthesize(self, text, output_path, voice=""):
        Path(output_path).write_bytes(b"ok")
        return output_path


def test_tts_router_uses_edge_like_fallback(tmp_path):
    router = FallbackTTSAdapter([("fish", _FailTTS()), ("edge", _GoodTTS())])
    output = tmp_path / "tts.mp3"
    assert router.synthesize("hello", str(output)) == str(output)
    assert output.read_bytes() == b"ok"


def test_tts_router_prioritizes_explicit_gemini_live_voice(tmp_path):
    calls = []

    class RecordingTTS:
        def __init__(self, name):
            self.name = name

        def synthesize(self, text, output_path, voice=""):
            calls.append(self.name)
            if self.name != "gemini-live":
                raise RuntimeError("wrong provider first")
            Path(output_path).write_bytes(b"live")
            return output_path

    router = FallbackTTSAdapter([
        ("fish", RecordingTTS("fish")),
        ("gemini-3.1-flash-live", RecordingTTS("gemini-live")),
        ("edge", RecordingTTS("edge")),
    ])
    output = tmp_path / "live.mp3"
    assert router.synthesize("xin chào", str(output), voice="gemini-live:Sulafat") == str(output)
    assert calls == ["gemini-live"]


def test_gemini_flash_live_voice_override_parser():
    assert GeminiFlashLiveTTSAdapter._voice_from_request("gemini-live:Sulafat") == "Sulafat"
    assert GeminiFlashLiveTTSAdapter._voice_from_request("fish:abc") is None


def test_elevenlabs_voice_override_and_free_default(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "configured-voice")
    assert ElevenLabsTTSAdapter._normalize_voice("elevenlabs:explicit-voice") == "explicit-voice"
    assert ElevenLabsTTSAdapter._normalize_voice("elevenlabs:env") == "configured-voice"
    assert ElevenLabsTTSAdapter._normalize_voice("vi-VN-NamMinhNeural") == "configured-voice"


def test_elevenlabs_pool_deduplicates_single_pool_and_numbered_keys(monkeypatch):
    for name in list(__import__("os").environ):
        if name.startswith("ELEVENLABS_API_KEY"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ELEVENLABS_API_KEYS", "key-a,key-b,key-a")
    monkeypatch.setenv("ELEVENLABS_API_KEY_2", "key-c")
    assert configured_elevenlabs_keys() == ["key-a", "key-b", "key-c"]
    adapter = ElevenLabsTTSAdapter()
    assert adapter.api_keys == ["key-a", "key-b", "key-c"]


def test_tts_router_prioritizes_explicit_elevenlabs_voice(tmp_path):
    calls = []

    class RecordingTTS:
        def __init__(self, name):
            self.name = name

        def synthesize(self, text, output_path, voice=""):
            calls.append((self.name, voice))
            if self.name != "elevenlabs":
                raise RuntimeError("wrong provider first")
            Path(output_path).write_bytes(b"eleven")
            return output_path

    router = FallbackTTSAdapter([
        ("fish", RecordingTTS("fish")),
        ("elevenlabs-v3", RecordingTTS("elevenlabs")),
        ("edge", RecordingTTS("edge")),
    ])
    output = tmp_path / "eleven.mp3"
    assert router.synthesize("xin chào", str(output), voice="elevenlabs:abc") == str(output)
    assert calls == [("elevenlabs", "elevenlabs:abc")]


class _FailASR:
    def transcribe(self, audio_path, language="auto"):
        raise RuntimeError("rate limited")


class _GoodASR:
    def transcribe(self, audio_path, language="auto"):
        return [SpeechSegment("S00000", 0.0, 1.0, "你好", "zh")]


def test_asr_router_uses_gemini_like_fallback():
    router = FallbackASRAdapter([("groq", _FailASR()), ("gemini", _GoodASR())])
    result = router.transcribe("ignored.wav", language="zh")
    assert len(result) == 1
    assert result[0].source_text == "你好"


def test_visual_pipeline_does_not_mux_audio():
    source = (Path(__file__).parents[1] / "src" / "features" / "visual" / "pipeline.py").read_text(encoding="utf-8")
    assert "mux_audio_video" not in source
    assert "src.infrastructure.video" not in source


def test_audio_pipeline_does_not_import_visual_features():
    source = (Path(__file__).parents[1] / "src" / "features" / "audio" / "pipeline.py").read_text(encoding="utf-8")
    assert "features.visual" not in source
    assert "Yolo" not in source


def test_asr_corrector_skips_secondary_when_primary_is_not_suspicious():
    class Secondary:
        def transcribe(self, audio_path, language="auto"):
            raise AssertionError("secondary ASR should not be called")

    segments = [SpeechSegment("S00000", 0.0, 1.0, "拜拜", "zh")]
    corrector = OpenRouterASRCorrector(
        api_keys=["k"], models=["m"], secondary_provider=Secondary()
    )
    result = corrector.correct_segments(segments, language="zh", audio_path="x.wav")
    assert result[0].source_text == "拜拜"


def test_asr_consensus_corrects_only_suspicious_text_and_preserves_timing(monkeypatch):
    class Secondary:
        def transcribe(self, audio_path, language="auto"):
            return [SpeechSegment("ALT", 4.9, 6.1, "有个外卖我送错。", "zh")]

    segments = [
        SpeechSegment("S00000", 0.0, 1.0, "拜拜", "zh"),
        SpeechSegment("S00001", 5.0, 6.0, "有個Wi-Fi我送錯了", "zh"),
    ]
    corrector = OpenRouterASRCorrector(
        api_keys=["k"], models=["m"], secondary_provider=Secondary()
    )
    monkeypatch.setattr(
        corrector,
        "_post",
        lambda payload, key: '{"corrections":[{"id":1,"text":"有个外卖我送错。"}]}',
    )
    result = corrector.correct_segments(segments, language="zh", audio_path="x.wav")
    assert result[0].source_text == "拜拜"
    assert result[1].source_text == "有个外卖我送错。"
    assert result[1].segment_id == "S00001"
    assert result[1].start_time == 5.0
    assert result[1].end_time == 6.0


def test_asr_consensus_retries_when_first_model_preserves_mixed_cjk_latin(monkeypatch):
    class Secondary:
        def transcribe(self, audio_path, language="auto"):
            return [SpeechSegment("ALT", 4.9, 6.1, "有个外卖我送错。", "zh")]

    segments = [SpeechSegment("S00001", 5.0, 6.0, "有個Wi-Fi我送錯了", "zh")]
    corrector = OpenRouterASRCorrector(
        api_keys=["k"], models=["m1", "m2"], secondary_provider=Secondary()
    )
    calls = []

    def fake_post(payload, key):
        calls.append(payload["model"])
        if payload["model"] == "m1":
            return '{"corrections":[{"id":0,"text":"有個Wi-Fi我送錯了"}]}'
        return '{"corrections":[{"id":0,"text":"有个外卖我送错。"}]}'

    monkeypatch.setattr(corrector, "_post", fake_post)
    result = corrector.correct_segments(segments, language="zh", audio_path="x.wav")
    assert calls == ["m1", "m2"]
    assert result[0].source_text == "有个外卖我送错。"
