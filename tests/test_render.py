import os
import pytest
import subprocess
from src.infrastructure.providers.video.ffmpeg_video_provider import FFmpegVideoProvider
from src.infrastructure.providers.video.mock_video_provider import MockVideoProvider
from src.application.use_cases.render_video import RenderVideoUseCase
from src.infrastructure.artifacts.artifact_manager import ArtifactManager
from src.shared.exceptions import ProviderError, ValidationError

# ------------------------------------------------------------------
# MockVideoProvider render tests
# ------------------------------------------------------------------
class TestMockRender:
    def test_mock_render_basic(self):
        provider = MockVideoProvider()
        result = provider.render_video("input.mp4", "input.srt")
        assert os.path.exists(result)
        assert os.path.basename(result) == "input_translated.mp4"
        os.remove(result)

    def test_mock_render_with_output_path(self):
        provider = MockVideoProvider()
        out = os.path.join(os.path.dirname(__file__), "_mock_render_test.mp4")
        try:
            result = provider.render_video("input.mp4", "input.srt", output_path=out)
            assert result == os.path.normpath(out)
            assert os.path.exists(result)
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_mock_render_with_style(self):
        provider = MockVideoProvider()
        result = provider.render_video("input.mp4", "input.srt", style={"font_size": 24})
        assert os.path.exists(result)
        assert os.path.basename(result) == "input_translated.mp4"
        os.remove(result)

# ------------------------------------------------------------------
# FFmpegVideoProvider render tests (no real ffmpeg)
# ------------------------------------------------------------------
class TestFFmpegRender:
    def test_render_missing_video(self, monkeypatch):
        def mock_run(*args, **kwargs):
            return subprocess.CompletedProcess([], 0)
        monkeypatch.setattr(subprocess, "run", mock_run)

        provider = FFmpegVideoProvider()
        with pytest.raises(ProviderError, match="Input video file not found"):
            provider.render_video("nonexistent_video.mp4", "sub.srt")

    def test_render_missing_subtitle(self, monkeypatch):
        def mock_run(*args, **kwargs):
            return subprocess.CompletedProcess([], 0)
        monkeypatch.setattr(subprocess, "run", mock_run)

        # Create dummy video file so we pass the video check
        video_path = os.path.join(os.path.dirname(__file__), "_dummy_video.mp4")
        try:
            with open(video_path, "w") as f:
                f.write("dummy")
            provider = FFmpegVideoProvider()
            with pytest.raises(ProviderError, match="Subtitle file not found"):
                provider.render_video(video_path, "nonexistent_sub.srt")
        finally:
            if os.path.exists(video_path):
                os.remove(video_path)

    def test_render_no_ffmpeg(self, monkeypatch):
        def mock_run(*args, **kwargs):
            raise FileNotFoundError()
        monkeypatch.setattr(subprocess, "run", mock_run)

        provider = FFmpegVideoProvider()
        with pytest.raises(ProviderError, match="FFmpeg is not installed"):
            provider.render_video("dummy.mp4", "dummy.srt")

# ------------------------------------------------------------------
# UseCase render tests
# ------------------------------------------------------------------
class TestRenderVideoUseCase:
    def test_use_case_basic(self):
        provider = MockVideoProvider()
        uc = RenderVideoUseCase(provider)
        result = uc.execute("input.mp4", "input.srt")
        assert os.path.exists(result)
        os.remove(result)

    def test_use_case_with_output_path(self):
        provider = MockVideoProvider()
        uc = RenderVideoUseCase(provider)
        out = os.path.join(os.path.dirname(__file__), "_uc_render_test.mp4")
        try:
            result = uc.execute("input.mp4", "input.srt", output_path=out)
            assert result == os.path.normpath(out)
            assert os.path.exists(result)
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_use_case_with_style(self):
        provider = MockVideoProvider()
        uc = RenderVideoUseCase(provider)
        result = uc.execute("input.mp4", "input.srt", style={"font_size": 24})
        assert os.path.exists(result)
        assert os.path.basename(result) == "input_translated.mp4"
        os.remove(result)

# ------------------------------------------------------------------
# ArtifactManager render path tests
# ------------------------------------------------------------------
class TestArtifactRenderPath:
    def test_video_output_path(self):
        mgr = ArtifactManager(base_dir="output")
        path = mgr.video_path("test_job")
        assert path == os.path.normpath("output/test_job/videos/output.mp4")

    def test_video_output_path_different_base(self):
        mgr = ArtifactManager(base_dir="renders")
        path = mgr.video_path("job_xyz")
        assert path == os.path.normpath("renders/job_xyz/videos/output.mp4")
