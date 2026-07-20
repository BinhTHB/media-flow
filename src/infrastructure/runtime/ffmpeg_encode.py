from __future__ import annotations

from dataclasses import dataclass

from src.domain.entities.resource_policy import FFmpegConfig
from src.infrastructure.runtime.ffmpeg_capability_detector import (
    FFmpegCapabilityReport,
)


@dataclass(frozen=True)
class FFmpegEncodePlan:
    args: tuple[str, ...]
    encoder_used: str
    gpu_decode_used: bool


def build_ffmpeg_encode_args(
    input_path: str,
    output_path: str,
    config: FFmpegConfig,
    has_cpu_filters: bool,
    capabilities: FFmpegCapabilityReport,
) -> FFmpegEncodePlan:
    use_nvenc = (
        config.encoder == "nvenc"
        or (
            config.encoder == "auto"
            and config.use_gpu_encode
            and capabilities.nvenc_available
        )
    ) and capabilities.nvenc_available
    gpu_decode = (
        config.use_gpu_decode
        and config.decoder == "cuda"
        and capabilities.cuda_hwaccel_available
        and not has_cpu_filters
    )
    args = ["ffmpeg"]
    if gpu_decode:
        args.extend(("-hwaccel", "cuda"))
    args.extend(("-i", input_path))
    if use_nvenc:
        codec = (
            config.nvenc_codec
            if config.nvenc_codec in capabilities.supported_nvenc_codecs
            else capabilities.supported_nvenc_codecs[0]
        )
        args.extend(
            (
                "-c:v", codec,
                "-preset", config.nvenc_preset,
                "-cq", str(config.nvenc_cq),
            )
        )
    else:
        codec = config.cpu_codec
        args.extend(
            (
                "-c:v", codec,
                "-preset", config.cpu_preset,
                "-crf", str(config.cpu_crf),
            )
        )
    args.extend(("-pix_fmt", config.pixel_format, "-c:a", "copy", "-y", output_path))
    return FFmpegEncodePlan(tuple(args), codec, gpu_decode)
