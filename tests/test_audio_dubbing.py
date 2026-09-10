from __future__ import annotations

import json
import shutil
import wave
from pathlib import Path

import pytest

from src.features.audio.composer import AudioComposer
from src.features.audio.models import TTSClip
from src.infrastructure.speech.groq_whisper import GroqWhisperASRAdapter


def _write_silence(path: Path, duration: float, rate: int = 16000) -> None:
    frames = max(1, int(round(duration * rate)))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * frames)


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="AudioComposer integration requires FFmpeg/FFprobe",
)
def test_audio_composer_fits_delays_and_preserves_total_duration(tmp_path: Path):
    clip_a = tmp_path / "a.wav"
    clip_b = tmp_path / "b.wav"
    _write_silence(clip_a, 0.25)
    _write_silence(clip_b, 0.60)

    output = tmp_path / "dubbed.wav"
    result = AudioComposer().compose_dubbed_timeline(
        tts_clips=[
            TTSClip("S1", 0.50, 1.00, str(clip_a), 0.50),
            TTSClip("S2", 1.50, 2.00, str(clip_b), 0.50),
        ],
        total_duration=2.50,
        output_audio_path=str(output),
    )

    assert Path(result).is_file()
    with wave.open(str(output), "rb") as handle:
        duration = handle.getnframes() / float(handle.getframerate())
    assert duration == pytest.approx(2.50, abs=0.03)


def test_audio_composer_never_slows_short_utterances(tmp_path: Path, monkeypatch):
    composer = AudioComposer(max_speedup=1.25)
    source = tmp_path / "short.wav"
    output = tmp_path / "fit.wav"
    source.write_bytes(b"fake")
    commands = []

    monkeypatch.setattr(composer, "probe_duration", lambda path: 0.60)
    monkeypatch.setattr(composer, "_run", lambda cmd, action: commands.append(cmd))
    composer._fit_clip(source, output, target_duration=1.0)

    filter_arg = commands[0][commands[0].index("-filter:a") + 1]
    assert "atempo=" not in filter_arg
    assert "apad=whole_dur=1.000000" in filter_arg


def test_audio_composer_rejects_extreme_speedup(tmp_path: Path, monkeypatch):
    composer = AudioComposer(max_speedup=1.25)
    source = tmp_path / "long.wav"
    output = tmp_path / "fit.wav"
    source.write_bytes(b"fake")
    monkeypatch.setattr(composer, "probe_duration", lambda path: 2.0)

    with pytest.raises(RuntimeError, match="above naturalness cap"):
        composer._fit_clip(source, output, target_duration=1.0)


def test_gemini_audio_asr_parses_canonical_segments_without_network(tmp_path: Path, monkeypatch):
    from src.infrastructure.speech import gemini_audio

    fake_audio = tmp_path / "chunk.mp3"
    fake_audio.write_bytes(b"fake-audio")

    class FakeResponse:
        text = json.dumps({
            "detected_language": "zh",
            "segments": [
                {"start": 0.2, "end": 1.1, "text": "你好"},
                {"start": 1.3, "end": 2.0, "text": "再见"},
            ],
        }, ensure_ascii=False)

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def generate_content(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(gemini_audio, "GeminiClient", FakeClient)
    adapter = gemini_audio.GeminiAudioASRAdapter(api_key="test")
    result = adapter._transcribe_chunk(
        fake_audio,
        language="auto",
        offset=5.0,
        index_base=7,
    )

    assert [segment.segment_id for segment in result] == ["S00007", "S00008"]
    assert result[0].start_time == pytest.approx(5.2)
    assert result[1].end_time == pytest.approx(7.0)
    assert result[0].source_language == "zh"
    assert result[0].source_text == "你好"


def test_groq_multipart_contains_model_and_audio(tmp_path: Path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF-test")
    body, content_type = GroqWhisperASRAdapter._multipart_body(
        {"model": "whisper-large-v3-turbo", "response_format": "verbose_json"},
        audio,
    )

    assert content_type.startswith("multipart/form-data; boundary=")
    assert b"whisper-large-v3-turbo" in body
    assert b"sample.wav" in body
    assert b"RIFF-test" in body


def test_groq_default_prefers_quality_large_v3():
    assert GroqWhisperASRAdapter(api_key="test").model == "whisper-large-v3"


def test_mux_wav_audio_forces_aac_for_mp4_compatibility(tmp_path: Path, monkeypatch):
    from src.infrastructure.video import ffmpeg as ffmpeg_utils

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        class Result:
            stdout = ""
        return Result()

    monkeypatch.setattr(ffmpeg_utils.subprocess, "run", fake_run)
    output = tmp_path / "final.mp4"
    ffmpeg_utils.mux_audio_video("visual.mp4", "dubbed.wav", str(output))

    assert len(calls) == 1
    command = calls[0]
    audio_codec_index = command.index("-c:a")
    assert command[audio_codec_index + 1] == "aac"
    assert "-b:a" in command
    assert "-shortest" in command
