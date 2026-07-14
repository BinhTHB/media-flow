import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.application.use_cases.generate_dubbed_audio import GenerateDubbedAudioUseCase
from src.domain.entities.segment import Segment
from src.domain.entities.tts_config import TTSConfig
from src.infrastructure.artifacts.artifact_manager import ArtifactManager
from src.infrastructure.providers.tts.ffmpeg_dubbing_provider import FFmpegDubbingProvider
from src.shared.exceptions import ProviderError


def test_generate_dubbed_audio_fails_on_any_failed_segment(tmp_path):
    artifacts = ArtifactManager(str(tmp_path))
    tts = Mock()
    calls = 0

    def synthesize(text, output_path, voice, rate):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ProviderError("offline")
        Path(output_path).write_bytes(b"audio")

    tts.synthesize.side_effect = synthesize
    dubbing = Mock()
    dubbing.merge_timeline.return_value = str(tmp_path / "dubbed_audio.wav")
    segments = [
        Segment("one", 0, 1, "one", "một"),
        Segment("two", 2, 3, "two", "hai"),
    ]

    with pytest.raises(ProviderError, match="TTS segment failed"):
        GenerateDubbedAudioUseCase(tts, dubbing, artifacts).execute(
            "job", segments, 3.0, TTSConfig()
        )

    manifest = json.loads(Path(artifacts.tts_manifest_path("job")).read_text("utf-8"))
    assert manifest["segments"][-1]["status"] == "failed"
    dubbing.merge_timeline.assert_not_called()


def test_generate_dubbed_audio_fails_when_every_segment_fails(tmp_path):
    artifacts = ArtifactManager(str(tmp_path))
    tts = Mock()
    tts.synthesize.side_effect = ProviderError("offline")
    dubbing = Mock()

    with pytest.raises(ProviderError, match="TTS segment failed"):
        GenerateDubbedAudioUseCase(tts, dubbing, artifacts).execute(
            "job", [Segment("one", 0, 1, "one", "một")], 1.0, TTSConfig()
        )


def test_ffmpeg_dubbing_commands_include_timeline_and_audio_policy(tmp_path, monkeypatch):
    provider = FFmpegDubbingProvider()
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        Path(command[-1]).parent.mkdir(parents=True, exist_ok=True)
        Path(command[-1]).write_bytes(b"x")

    monkeypatch.setattr("subprocess.run", run)
    monkeypatch.setattr(provider, "_probe_duration", lambda path: 2.5)
    monkeypatch.setattr(provider, "_probe_audio_streams", lambda path: [{"index": 1}])
    clip = tmp_path / "one.mp3"
    clip.write_bytes(b"x")
    provider.merge_timeline(
        [{"audio_path": str(clip), "start": 1.25, "end": 2.25}],
        str(tmp_path / "dubbed.wav"),
        4.0,
    )
    provider.render_dubbed_video(
        "video.mp4", "dubbed.wav", str(tmp_path / "dubbed.mp4"), "lower", 0.2
    )

    assert "adelay=1250|1250" in commands[0][commands[0].index("-filter_complex") + 1]
    assert "atempo=2.000000,atempo=1.250000" in commands[0][commands[0].index("-filter_complex") + 1]
    assert "atrim=duration=1.000" in commands[0][commands[0].index("-filter_complex") + 1]
    assert "apad=pad_dur=4.0" in commands[0][commands[0].index("-filter_complex") + 1]
    assert "volume=0.2" in commands[1][commands[1].index("-filter_complex") + 1]

def test_replace_mux_maps_only_dubbed_audio_and_rejects_multiple_streams(tmp_path, monkeypatch):
    provider = FFmpegDubbingProvider()
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        Path(command[-1]).parent.mkdir(parents=True, exist_ok=True)
        Path(command[-1]).write_bytes(b"x")

    monkeypatch.setattr("subprocess.run", run)
    monkeypatch.setattr(provider, "_probe_audio_streams", lambda path: [{"index": 1}])

    provider.render_dubbed_video(
        "video.mp4", "dubbed.wav", str(tmp_path / "replaced.mp4"), "replace", 0.2
    )

    assert commands[0][commands[0].index("-map") + 1] == "0:v:0"
    assert commands[0][commands[0].index("-map", commands[0].index("-map") + 1) + 1] == "1:a:0"
    assert "-filter_complex" not in commands[0]

def test_merge_timeline_rejects_overlapping_target_ranges(tmp_path):
    provider = FFmpegDubbingProvider()
    first = tmp_path / "one.mp3"
    second = tmp_path / "two.mp3"
    first.write_bytes(b"x")
    second.write_bytes(b"x")

    with pytest.raises(ProviderError, match="overlapping clips"):
        provider.merge_timeline(
            [
                {"audio_path": str(first), "start": 0.0, "end": 2.0},
                {"audio_path": str(second), "start": 1.0, "end": 3.0},
            ],
            str(tmp_path / "dubbed.wav"),
            3.0,
        )


def test_generate_dubbed_audio_fails_for_zero_byte_segment(tmp_path):
    artifacts = ArtifactManager(str(tmp_path))
    tts = Mock()
    dubbing = Mock()

    with pytest.raises(ProviderError, match="missing or empty"):
        GenerateDubbedAudioUseCase(tts, dubbing, artifacts).execute(
            "job", [Segment("one", 0, 1, "one", "một")], 1.0, TTSConfig()
        )

    dubbing.merge_timeline.assert_not_called()
