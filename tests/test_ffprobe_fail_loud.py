import pytest
from src.infrastructure.benchmark.hybrid_subtitle.video_utils import ffprobe_metadata

def test_ffprobe_metadata_fails_loud_on_error(monkeypatch):
    import subprocess
    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["ffprobe"])
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="CRITICAL-FAIL: ffprobe metadata extraction failed"):
        ffprobe_metadata("nonexistent.mp4")
