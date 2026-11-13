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
        return [f"Đã dịch: {text}" for text in texts]


def test_translation_router_uses_fallback():
    router = FallbackTranslationAdapter([
        ("primary", _FailTranslation()),
        ("fallback", _GoodTranslation()),
    ])
    assert router.translate_texts(["a", "b"]) == ["Đã dịch: a", "Đã dịch: b"]


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


def test_validator_accepts_proper_token_shared_across_source_lines():
    # Real regression: two "Douyin account: Spongecake" captions next to a
    # "@抖音: Spongecake" line were permanently rejected because the proper
    # token registered by the CJK+Latin line looked "leaked" into the
    # pure-Latin lines that legitimately contain the same name.
    source = [
        "Douyin account: Spongecake",
        "@抖音: Spongecake",
        "Douyin account: Spongecake",
    ]
    translated = [
        "Tài khoản Douyin: Spongecake",
        "@Douyin: Spongecake",
        "Tài khoản Douyin: Spongecake",
    ]

    assert FallbackTranslationAdapter._suspicious_indices(source, translated) == []


def test_validator_still_flags_token_migration_into_unrelated_line():
    source = ["Bloomberg终端上", "今天天气很好"]
    translated = ["Thị trường hôm nay rất đẹp", "Bloomberg terminal đã báo cáo"]

    assert 1 in FallbackTranslationAdapter._suspicious_indices(source, translated)


def test_validator_flags_prompt_boilerplate_echo():
    # Real gemma-4-26b output: the model answered with task instructions
    # instead of a translation, and every other check was blind to it
    # (non-empty, no numbers, no Latin protected tokens, CJK-free).
    source = ["One: be gentle", "轻声细语"]
    translated = [
        "Preserve line numbers (`i: translated_text`).",
        "Dịch theo định dạng numbered lines nhé",
    ]

    assert FallbackTranslationAdapter._suspicious_indices(source, translated) == [0, 1]


def test_validator_flags_untranslated_cjk_echo_with_gloss():
    # Real gemma-4-26b "explain mode" output: source copied verbatim plus an
    # English gloss. _normalize strips CJK so the echo itself was invisible
    # to every earlier check and rendered as a READY "translation".
    source = ["妈妈总是教导我们", "一位合格的淑女必须遵循以下规则"]
    translated = [
        "妈妈总是教导我们 (Chinese: Mom always teaches us)",
        "A qualified lady must follow these rules (English)",
    ]

    assert FallbackTranslationAdapter._suspicious_indices(source, translated) == [0, 1]


def test_validator_accepts_real_vietnamese_of_cjk_source():
    # Guard against false positives: clean Vietnamese contains no CJK at all.
    source = ["妈妈总是教导我们", "轻声细语"]
    translated = ["Mẹ luôn dạy chúng con", "Nói khẽ nói nhẹ"]

    assert FallbackTranslationAdapter._suspicious_indices(source, translated) == []


def test_validator_flags_describe_mode_english_for_vi_target():
    # Real gemma-4-26b "describe mode" output: numbered lines with English
    # descriptions instead of Vietnamese translations. No CJK echo, no gloss
    # parentheses — only the missing-diacritics check can catch these.
    source = ["眼神柔和", "双手轻柔", "轻声细语"]
    translated = [
        'Description: "Soft eyes/gaze"',
        'Description: "Hands gentle/softly"',
        'Instruction: "Be gentle"',
    ]

    assert FallbackTranslationAdapter._suspicious_indices(source, translated) == [0, 1, 2]


def test_validator_allows_short_proper_noun_passthrough_for_vi_target():
    source = ["Spongecake", "WiFi 6"]
    translated = ["Spongecake", "WiFi 6"]

    assert FallbackTranslationAdapter._suspicious_indices(source, translated) == []


def test_validator_flags_ordinal_copied_from_neighbor_rule():
    source = ["妈妈总是教导我们", "第一条: 要温柔"]
    translated = ["Thứ nhất: Hãy dịu dàng", "Thứ nhất: Phải dịu dàng"]

    assert 0 in FallbackTranslationAdapter._suspicious_indices(source, translated)
    assert 1 not in FallbackTranslationAdapter._suspicious_indices(source, translated)


def test_validator_flags_untranslated_long_english_source_for_vi_target():
    # An English sentence "translated" as itself is not a proper-noun
    # passthrough; it must be flagged so the fail-safe suppresses it.
    source = ["A qualified lady must follow these rules"]
    translated = ["A qualified lady must follow these rules"]

    assert FallbackTranslationAdapter._suspicious_indices(source, translated) == [0]


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


def test_tts_router_stops_calling_a_provider_only_after_zero_credits(tmp_path):
    state = {"left": 69}
    calls = []

    class Eleven:
        def synthesize(self, text, output_path, voice=""):
            calls.append("eleven")
            left = state["left"]
            state["left"] = 0
            if left == 0:
                raise RuntimeError("quota_exceeded You have 0 credits remaining")
            raise RuntimeError(
                "quota_exceeded You have 69 credits remaining, while 98 credits are required"
            )

    class Edge:
        def synthesize(self, text, output_path, voice=""):
            calls.append("edge")
            Path(output_path).write_bytes(b"ok")
            return output_path

    router = FallbackTTSAdapter([("elevenlabs-v3", Eleven()), ("edge-tts", Edge())])
    for name in ("a", "b", "c"):
        router.synthesize("xin chào", str(tmp_path / f"{name}.mp3"), voice="elevenlabs:brian")
    assert calls == ["eleven", "edge", "eleven", "edge", "edge"]


def test_edge_tts_retries_when_the_service_returns_no_audio(tmp_path, monkeypatch):
    import sys
    import types

    from src.infrastructure.tts.edge_tts import EdgeTTSAdapter

    calls = {"n": 0}

    class Communicate:
        def __init__(self, text, voice, rate, volume, pitch, **_kwargs):
            self.text = text

        async def save(self, path):
            calls["n"] += 1
            if calls["n"] < 2:
                raise RuntimeError(
                    "No audio was received. Please verify that your parameters are correct."
                )
            Path(path).write_bytes(b"ID3audio")

    fake = types.ModuleType("edge_tts")
    fake.Communicate = Communicate
    monkeypatch.setitem(sys.modules, "edge_tts", fake)
    output = tmp_path / "line.mp3"
    assert EdgeTTSAdapter().synthesize("Rắc muối trắng xóa lên từng nửa quả.", str(output)) == str(output)
    assert calls["n"] == 2
    assert output.stat().st_size > 0


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


_COLLAPSED_ASR = (
    "b n Ngay gi c v h l b d k nh t b tr d t t r mu tr x l t n qu "
    "Mu r s n gi cho c kh th M tu sau t tr cao nh xu c c th m t th"
)
_READABLE_REPAIR = "Ông mang cà chua ra phơi nắng đến khi đỏ rực, mười cân tươi chỉ còn một cân."


def test_collapsed_asr_is_replaced_from_secondary_without_a_language_model():
    class Secondary:
        def transcribe(self, audio_path, language="auto"):
            return [SpeechSegment("ALT", 64.8, 81.0, _READABLE_REPAIR, "vi")]

    class AlsoCollapsed:
        def transcribe(self, audio_path, language="auto"):
            return [SpeechSegment("ALT", 64.8, 81.0, _COLLAPSED_ASR, "vi")]

    segments = [SpeechSegment("S00014", 64.8, 81.07, _COLLAPSED_ASR, "Vietnamese")]
    repaired = OpenRouterASRCorrector(
        api_keys=[], models=[], secondary_provider=Secondary()
    ).correct_segments(segments, language="vi", audio_path="x.wav")
    assert repaired[0].source_text == _READABLE_REPAIR
    assert repaired[0].segment_id == "S00014"
    assert repaired[0].start_time == 64.8
    assert repaired[0].end_time == 81.07

    untouched = [SpeechSegment("S00014", 64.8, 81.07, _COLLAPSED_ASR, "Vietnamese")]
    kept = OpenRouterASRCorrector(
        api_keys=[], models=[], secondary_provider=AlsoCollapsed()
    ).correct_segments(untouched, language="vi", audio_path="x.wav")
    assert kept[0].source_text == _COLLAPSED_ASR


def test_collapsed_window_keeps_each_secondary_line_on_its_own_time():
    class Secondary:
        def transcribe(self, audio_path, language="auto"):
            return [
                SpeechSegment("A", 64.9, 70.0, "Ông bán ánh nắng.", "vi"),
                SpeechSegment("B", 70.2, 76.0, "Cà vừa hái là bổ đôi.", "vi"),
                SpeechSegment("C", 76.2, 81.0, "Cánh đồng thành một tấm thảm đỏ rực.", "vi"),
            ]

    segments = [
        SpeechSegment("S00013", 58.98, 64.8, "Vụ thứ hai, ông không bán cà chua nữa,", "vi"),
        SpeechSegment("S00014", 64.8, 81.07, _COLLAPSED_ASR, "Vietnamese"),
    ]
    repaired = OpenRouterASRCorrector(
        api_keys=[], models=[], secondary_provider=Secondary()
    ).correct_segments(segments, language="vi", audio_path="x.wav")
    texts = [segment.source_text for segment in repaired]
    assert texts[0] == "Vụ thứ hai, ông không bán cà chua nữa,"
    assert texts[1:] == [
        "Ông bán ánh nắng.",
        "Cà vừa hái là bổ đôi.",
        "Cánh đồng thành một tấm thảm đỏ rực.",
    ]
    assert repaired[1].start_time == pytest.approx(64.9)
    assert repaired[2].start_time == pytest.approx(70.2)
    assert repaired[-1].end_time == pytest.approx(81.0)


def test_defective_line_is_replaced_and_a_shared_name_is_kept():
    class Secondary:
        def transcribe(self, audio_path, language="auto"):
            return [
                SpeechSegment("A", 123.0, 125.4, "bánh pizza ở Ý", "vi"),
                SpeechSegment("B", 10.9, 14.1, "Ông tên Mehmet, một nông dân nghèo.", "vi"),
            ]

    segments = [
        SpeechSegment("S00030", 123.1, 125.3, "bánh tisa ở Ý", "Vietnamese"),
        SpeechSegment("S00003", 10.9, 14.1, "Ông tên Mehmet, một nông dân nghèo.", "Vietnamese"),
    ]
    repaired = OpenRouterASRCorrector(
        api_keys=[], models=[], secondary_provider=Secondary()
    ).correct_segments(segments, language="vi", audio_path="x.wav")
    assert repaired[0].source_text == "bánh pizza ở Ý"
    assert repaired[0].start_time == 123.1
    assert repaired[1].source_text == "Ông tên Mehmet, một nông dân nghèo."


def test_second_read_fixes_a_loanword_without_pasting_the_paragraph():
    class Secondary:
        def transcribe(self, audio_path, language="auto"):
            return [
                SpeechSegment("A", 120.0, 122.5, "có mặt trên bánh pizza ở Ý.", "vi"),
                SpeechSegment("B", 123.2, 127.2, "Trong lọ dầu ô liu ở Đức, trong chiếc bơ gờ ở Mỹ.", "vi"),
                SpeechSegment("C", 10.9, 14.1, "Ông tên mê mét, một nông dân nghèo.", "vi"),
            ]

    segments = [
        SpeechSegment("S00036", 123.1, 125.3, "bánh tisa ở Ý trong lọ dầu", "Vietnamese"),
        SpeechSegment("S00037", 125.3, 127.1, "trong chiếc burger ở Mỹ.", "Vietnamese"),
        SpeechSegment("S00003", 10.9, 14.1, "Ông tên Mehmet, một nông dân nghèo.", "Vietnamese"),
    ]
    repaired = OpenRouterASRCorrector(
        api_keys=[], models=[], secondary_provider=Secondary()
    ).correct_segments(segments, language="vi", audio_path="x.wav")
    assert repaired[0].source_text == "có mặt trên bánh pizza ở Ý."
    assert repaired[0].start_time == 123.1
    assert repaired[0].end_time == 125.3
    assert repaired[1].source_text == "trong chiếc burger ở Mỹ."
    assert repaired[2].source_text == "Ông tên Mehmet, một nông dân nghèo."


def test_defective_line_stays_when_the_second_read_fails():
    class Secondary:
        def transcribe(self, audio_path, language="auto"):
            raise RuntimeError("gemini down")

    segments = [SpeechSegment("S00030", 123.1, 125.3, "bánh tisa ở Ý", "Vietnamese")]
    kept = OpenRouterASRCorrector(
        api_keys=[], models=[], secondary_provider=Secondary()
    ).correct_segments(segments, language="vi", audio_path="x.wav")
    assert kept[0].source_text == "bánh tisa ở Ý"


def test_mock_speech_label_still_attaches_gemini_repair(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    from src.interfaces.composition.providers import build_asr_corrector

    corrector = build_asr_corrector("auto", "mock")
    assert corrector is not None
    assert corrector.secondary_provider is not None
    assert corrector.secondary_provider.model == "gemini-3.5-flash-lite"


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


def test_validator_accepts_latin_canonical_character_names():
    # Real gemma output on a One Piece clip: CJK person names are canonically
    # rendered as Latin canonical names without Vietnamese diacritics. The
    # missing-diacritics check used to reject every one of them, killing the
    # whole chunk they lived in.
    source = [
        "巴索罗米·熊",
        "月光·莫利亚",
        "乔拉可尔·米霍克",
        "波雅·汉库克",
        "唐吉诃德·多弗朗明哥",
    ]
    translated = [
        "Bartholomew Kuma",
        "Gekko Moria",
        "Dracule Mihawk",
        "Boa Hancock",
        "Donquixote Doflamingo",
    ]

    assert FallbackTranslationAdapter._suspicious_indices(source, translated) == []


def test_validator_accepts_name_shaped_output_without_source_dot():
    source = ["海贼王", "AI制作"]
    translated = ["One Piece", "AI"]

    assert FallbackTranslationAdapter._suspicious_indices(source, translated) == []


def test_validator_still_flags_english_gloss_without_diacritics():
    # Name-shaped passthrough must not open the door to describe-mode
    # English: sentence words and lowercase words still fail.
    source = ["眼神柔和", "AI制作"]
    translated = ["Soft eyes", "Made with AI"]

    assert FallbackTranslationAdapter._suspicious_indices(source, translated) == [0, 1]


def test_router_fails_only_unresolved_lines_not_the_whole_chunk():
    # One poisoned line used to raise and mark every neighbor in the chunk
    # TRANSLATION_FAILED. The router must return best-effort results and
    # leave only the poisoned line empty (caller maps "" to TRANSLATION_EMPTY).
    class PoisonedMiddle:
        def translate_texts(self, texts, target_language="vi", source_language="auto"):
            return [
                "Preserve line numbers (`i: translated_text`)."
                if "hai" in text
                else f"Đã dịch: {text}"
                for text in texts
            ]

    router = FallbackTranslationAdapter([("gemini", PoisonedMiddle())])
    result = router.translate_texts(["một", "hai", "ba"], target_language="vi")
    assert result == ["Đã dịch: một", "", "Đã dịch: ba"]


def test_ocr_wiring_is_local_rapidocr(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "not-used-for-ocr")
    monkeypatch.setenv("MEDIAFLOW_OCR_PROVIDER", "gemini")
    from src.interfaces.composition.providers import build_ocr_provider

    provider = build_ocr_provider("gemini")
    assert provider.model == "rapidocr"


def test_mirrored_caption_box_stays_inside_the_frame():
    from src.core.models.bbox import BBox
    from src.features.visual.pipeline import mirror_bbox

    box = BBox(10, 20, 40, 50, frame_width=100, frame_height=80)
    flipped = mirror_bbox(box)
    assert (flipped.x1, flipped.y1, flipped.x2, flipped.y2) == (60, 20, 90, 50)


def test_rapid_ocr_reads_txts_from_the_engine():
    import numpy as np

    from src.infrastructure.ocr.rapid_ocr import RapidOCRAdapter

    class Engine:
        def __call__(self, crop, **_kwargs):
            class Out:
                txts = ("你好",)

            return Out()

    crop = np.zeros((8, 8, 3), dtype=np.uint8)
    assert RapidOCRAdapter(engine=Engine()).read_crops([crop, None]) == ["你好", ""]


def test_rapid_ocr_retries_an_empty_crop_with_the_detector():
    import numpy as np

    from src.infrastructure.ocr.rapid_ocr import RapidOCRAdapter

    calls = []

    class Engine:
        def __call__(self, crop, use_det=False, **_kwargs):
            calls.append(use_det)

            class Out:
                txts = ("从港口可以看见",) if use_det else ()

            return Out()

    crop = np.zeros((8, 32, 3), dtype=np.uint8)
    assert RapidOCRAdapter(engine=Engine()).read_crops([crop]) == ["从港口可以看见"]
    assert calls == [False, True]


def test_fallback_ocr_only_sends_empty_crops_to_gemini():
    from src.infrastructure.ocr.router import FallbackOCRAdapter

    class Local:
        model = "rapidocr"

        def read_crops(self, crops):
            return ["已读", ""]

    class Cloud:
        model = "gemini"

        def __init__(self):
            self.seen = 0

        def read_crops(self, crops):
            self.seen = len(crops)
            return ["云端"]

    cloud = Cloud()
    texts = FallbackOCRAdapter(Local(), cloud).read_crops(["a", "b"])
    assert texts == ["已读", "云端"]
    assert cloud.seen == 1


def test_fallback_ocr_caps_how_many_empty_crops_go_to_gemini():
    from src.infrastructure.ocr.router import FallbackOCRAdapter

    class Local:
        model = "rapidocr"

        def read_crops(self, crops):
            return [""] * len(crops)

    class Cloud:
        model = "gemini"

        def __init__(self):
            self.seen = 0

        def read_crops(self, crops):
            self.seen = len(crops)
            return ["x"] * len(crops)

    cloud = Cloud()
    texts = FallbackOCRAdapter(Local(), cloud, max_fallback=2).read_crops(["a", "b", "c"])
    assert cloud.seen == 2
    assert texts == ["x", "x", ""]


def test_rapid_ocr_downscales_wide_1080p_plates():
    import numpy as np

    from src.infrastructure.ocr.rapid_ocr import RapidOCRAdapter

    crop = np.zeros((200, 1920, 3), dtype=np.uint8)
    fitted = RapidOCRAdapter._fit_crop(crop, max_edge=960)
    assert fitted.shape[1] == 960
    assert fitted.shape[0] == 100
