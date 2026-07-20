from __future__ import annotations

from dataclasses import asdict, dataclass
import subprocess


@dataclass(frozen=True)
class FFmpegCapabilityReport:
    ffmpeg_available: bool
    cuda_hwaccel_available: bool
    nvenc_available: bool
    supported_nvenc_codecs: tuple[str, ...]
    gpu_filters_available: tuple[str, ...]
    recommended_encode_mode: str

    def to_dict(self) -> dict:
        return asdict(self)


def _ffmpeg_output(*args: str) -> str:
    result = subprocess.run(
        ["ffmpeg", *args],
        capture_output=True,
        check=True,
        text=True,
        timeout=15,
    )
    return (result.stdout or "") + (result.stderr or "")


def detect_ffmpeg_capabilities() -> FFmpegCapabilityReport:
    try:
        hwaccels = _ffmpeg_output("-hide_banner", "-hwaccels").lower()
        encoders = _ffmpeg_output("-hide_banner", "-encoders").lower()
        filters = _ffmpeg_output("-hide_banner", "-filters").lower()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return FFmpegCapabilityReport(False, False, False, (), (), "cpu")

    codecs = tuple(
        codec for codec in ("h264_nvenc", "hevc_nvenc", "av1_nvenc")
        if codec in encoders
    )
    gpu_filters = tuple(
        name for name in ("scale_cuda", "cuda")
        if name in filters
    )
    return FFmpegCapabilityReport(
        ffmpeg_available=True,
        cuda_hwaccel_available="cuda" in hwaccels,
        nvenc_available=bool(codecs),
        supported_nvenc_codecs=codecs,
        gpu_filters_available=gpu_filters,
        recommended_encode_mode="nvenc" if codecs else "cpu",
    )
