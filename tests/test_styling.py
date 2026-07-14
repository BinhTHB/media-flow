import os
import subprocess

import pytest

from src.application.services.ass_generator import ASSGenerator
from src.application.use_cases.generate_subtitle import GenerateSubtitleUseCase
from src.domain.entities.segment import Segment
from src.domain.entities.subtitle_style import SubtitleStyle
from src.domain.entities.subtitle_style_preset import SubtitleStylePreset
from src.infrastructure.artifacts.artifact_manager import ArtifactManager
from src.infrastructure.providers.subtitle.ass_subtitle_provider import ASSSubtitleProvider
from src.infrastructure.providers.video.ffmpeg_video_provider import FFmpegVideoProvider
from src.shared.exceptions import ValidationError, UseCaseError


def _segments():
    return [
        Segment(id="s1", start=0.0, end=1.25, text="Hello", translated_text="Xin chào", source="speech", confidence=0.9),
        Segment(id="s2", start=2.0, end=3.5, text="World", translated_text="Thế giới", source="speech", confidence=0.9),
    ]


def test_subtitle_style_defaults_and_validation():
    style = SubtitleStyle()
    assert style.font_family == "Arial"
    assert style.font_size == 24
    assert style.primary_color == "#FFFFFF"

    with pytest.raises(ValidationError):
        SubtitleStyle(font_size=0)

    with pytest.raises(ValidationError):
        SubtitleStyle(alignment=10)


def test_subtitle_style_presets():
    youtube = SubtitleStylePreset.get("youtube")
    assert youtube.bold is True
    assert youtube.outline_width == 3.0

    custom = SubtitleStylePreset.get("youtube", {"font_size": 40})
    assert custom.font_size == 40

    with pytest.raises(ValidationError):
        SubtitleStylePreset.get("missing")


def test_ass_generator_outputs_valid_sections_and_dialogue():
    style = SubtitleStylePreset.get("netflix")
    content = ASSGenerator.generate(_segments(), style)
    assert "[Script Info]" in content
    assert "[V4+ Styles]" in content
    assert "[Events]" in content
    assert "Style: Default,Arial,28" in content
    assert "Dialogue: 0,0:00:00.00,0:00:01.25,Default" in content
    assert "Xin chào" in content


def test_ass_provider_and_use_case_generate_ass():
    provider = ASSSubtitleProvider()
    use_case = GenerateSubtitleUseCase(provider)
    content = use_case.execute(_segments(), subtitle_format="ass", style=SubtitleStylePreset.get("minimal"))
    assert "Style: Default" in content
    assert "Dialogue:" in content


def test_generate_subtitle_use_case_rejects_unknown_format():
    provider = ASSSubtitleProvider()
    use_case = GenerateSubtitleUseCase(provider)
    with pytest.raises(UseCaseError):
        use_case.execute(_segments(), subtitle_format="vtt")


def test_artifact_manager_subtitle_ass_path():
    mgr = ArtifactManager(base_dir="output")
    assert mgr.subtitle_ass_path("job1") == os.path.normpath("output/job1/subtitles/subtitle.ass")


def test_ffmpeg_uses_the_explicit_ass_path(monkeypatch, tmp_path):
    calls = []

    def mock_run(cmd, **kwargs):
        if cmd[:2] == ["ffmpeg", "-version"]:
            return subprocess.CompletedProcess(cmd, 0)
        calls.append(cmd)
        with open(cmd[-1], "w", encoding="utf-8") as f:
            f.write("rendered")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", mock_run)

    video = tmp_path / "input.mp4"
    srt = tmp_path / "subtitle.srt"
    ass = tmp_path / "subtitle.ass"
    out = tmp_path / "output.mp4"
    video.write_text("video", encoding="utf-8")
    srt.write_text("srt", encoding="utf-8")
    ass.write_text("ass", encoding="utf-8")

    provider = FFmpegVideoProvider()
    provider.render_video(str(video), str(ass), str(out))

    assert calls
    assert calls[-1][4].startswith("ass=")
    assert "subtitle.ass" in calls[-1][4]


def test_ffmpeg_falls_back_to_srt(monkeypatch, tmp_path):
    calls = []

    def mock_run(cmd, **kwargs):
        if cmd[:2] == ["ffmpeg", "-version"]:
            return subprocess.CompletedProcess(cmd, 0)
        calls.append(cmd)
        with open(cmd[-1], "w", encoding="utf-8") as f:
            f.write("rendered")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", mock_run)

    video = tmp_path / "input.mp4"
    srt = tmp_path / "subtitle.srt"
    out = tmp_path / "output.mp4"
    video.write_text("video", encoding="utf-8")
    srt.write_text("srt", encoding="utf-8")

    provider = FFmpegVideoProvider()
    provider.render_video(str(video), str(srt), str(out))

    assert calls
    assert calls[-1][4].startswith("subtitles=")
    assert "subtitle.srt" in calls[-1][4]
