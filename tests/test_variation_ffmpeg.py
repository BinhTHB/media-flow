"""Integration test for the ffmpeg variation adapter.

Opt-in: needs ffmpeg in PATH and a real source clip, so it is skipped by
default like the other media E2E tests. Enable with:

    $env:MEDIA_FLOW_RUN_MEDIA_INTEGRATION="1"
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from src.core.models.variation import VariationProfile
from src.infrastructure.video.variation import FFmpegVariationAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _have_ffmpeg() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


pytestmark = pytest.mark.skipif(
    os.getenv("MEDIA_FLOW_RUN_MEDIA_INTEGRATION") != "1" or not _have_ffmpeg(),
    reason="media integration tests are opt-in and need ffmpeg in PATH",
)


@pytest.fixture(scope="module")
def source_video(tmp_path_factory) -> str:
    """Synthesize a tiny 1080p clip so the test never depends on repo media."""
    out = tmp_path_factory.mktemp("media") / "source.mp4"
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc=size=1920x1080:rate=30:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return str(out)


def _probe(path: str) -> dict:
    from src.infrastructure.video.ffmpeg import probe_video

    return probe_video(path)


def test_identity_profile_copies_without_reencoding(source_video, tmp_path):
    adapter = FFmpegVariationAdapter()
    out = str(tmp_path / "identity.mp4")

    adapter.apply(source_video, VariationProfile(), out)

    # Byte-for-byte copy: same size, no encoder pass.
    assert Path(out).stat().st_size == Path(source_video).stat().st_size


def test_crop_keeps_source_resolution(source_video, tmp_path):
    adapter = FFmpegVariationAdapter()
    out = str(tmp_path / "crop.mp4")

    adapter.apply(source_video, VariationProfile(crop_ratio=0.03), out)

    info = _probe(out)
    assert (info["width"], info["height"]) == (1920, 1080)


def test_balanced_profile_produces_a_different_file(source_video, tmp_path):
    adapter = FFmpegVariationAdapter()
    out = str(tmp_path / "balanced.mp4")

    adapter.apply(
        source_video,
        VariationProfile(speed=1.03, crop_ratio=0.03, mirror=True, pitch_semitones=0.4),
        out,
    )

    info = _probe(out)
    assert (info["width"], info["height"]) == (1920, 1080)
    # Speed change shortens the clip; that is the point of the variation.
    assert info["duration"] < _probe(source_video)["duration"]
    assert Path(out).read_bytes() != Path(source_video).read_bytes()


def test_rotation_keeps_even_dimensions(source_video, tmp_path):
    adapter = FFmpegVariationAdapter()
    out = str(tmp_path / "rotate.mp4")

    adapter.apply(source_video, VariationProfile(rotate_degrees=1.5), out)

    info = _probe(out)
    assert info["width"] % 2 == 0
    assert info["height"] % 2 == 0


def test_adapter_raises_when_source_is_missing(tmp_path):
    adapter = FFmpegVariationAdapter()

    with pytest.raises(FileNotFoundError):
        adapter.apply(str(tmp_path / "nope.mp4"), VariationProfile(speed=1.02), str(tmp_path / "o.mp4"))
