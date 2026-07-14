from dataclasses import dataclass

from src.shared.exceptions import ValidationError


@dataclass(frozen=True)
class FFmpegConfig:
    encoder: str = "auto"
    decoder: str = "auto"
    use_gpu_encode: bool = True
    use_gpu_decode: bool = False
    nvenc_codec: str = "h264_nvenc"
    nvenc_preset: str = "p5"
    nvenc_cq: int = 23
    cpu_codec: str = "libx264"
    cpu_preset: str = "medium"
    cpu_crf: int = 23
    pixel_format: str = "yuv420p"

    def __post_init__(self) -> None:
        if self.encoder not in {"auto", "cpu", "nvenc"}:
            raise ValidationError("FFmpeg encoder must be auto, cpu, or nvenc.")
        if self.decoder not in {"auto", "cpu", "cuda"}:
            raise ValidationError("FFmpeg decoder must be auto, cpu, or cuda.")
        if not 0 <= self.nvenc_cq <= 51 or not 0 <= self.cpu_crf <= 51:
            raise ValidationError("FFmpeg quality values must be between 0 and 51.")


@dataclass(frozen=True)
class ResourcePolicy:
    profile: str = "balanced"
    allow_inpaint_light: bool = False
    allow_full_frame_segmentation: bool = False

    def __post_init__(self) -> None:
        if self.profile not in {"low", "balanced", "quality"}:
            raise ValidationError("Resource profile must be low, balanced, or quality.")

    @property
    def watermark_mode(self) -> str:
        return "gaussian" if self.profile == "low" else "blur_fill_feathered"

    @property
    def sample_interval_multiplier(self) -> float:
        return {"low": 2.0, "balanced": 1.0, "quality": 0.5}[self.profile]

    def can_inpaint_watermark(self, bbox_area_ratio: float) -> bool:
        return (
            self.profile == "quality"
            and self.allow_inpaint_light
            and bbox_area_ratio < 0.05
        )

    @property
    def expensive_validation_enabled(self) -> bool:
        return self.profile == "quality"
