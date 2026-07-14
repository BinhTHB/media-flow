import os
import pytest
import subprocess
from src.infrastructure.providers.video.ffmpeg_video_provider import FFmpegVideoProvider
from src.shared.exceptions import ProviderError

def test_ffmpeg_not_found(monkeypatch):
    def mock_run(*args, **kwargs):
        raise FileNotFoundError()
    monkeypatch.setattr(subprocess, "run", mock_run)

    provider = FFmpegVideoProvider()
    with pytest.raises(ProviderError, match="FFmpeg is not installed"):
        provider.extract_audio("dummy.mp4")

def test_ffmpeg_extract_no_video():
    provider = FFmpegVideoProvider()
    with pytest.raises(ProviderError, match="Input video file not found"):
        provider.extract_audio("nonexistent_video.mp4")
