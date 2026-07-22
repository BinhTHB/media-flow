from dataclasses import dataclass, field
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from src.domain.entities.placement_config import PlacementConfig
from src.domain.entities.resource_policy import FFmpegConfig, ResourcePolicy
from src.domain.entities.subtitle_quality_config import SubtitleQualityConfig
from src.domain.entities.subtitle_style import SubtitleStyle

@dataclass
class RuntimeConfig:
    keep_audio: bool = False
    frame_interval: float = 1.0
    subtitle_format: str = "srt"
    subtitle_style: str = "default"
    disable_segmentation: bool = False

@dataclass
class OutputConfig:
    output_dir: str = "output"

@dataclass
class SubtitleQualityConfigBinding:
    quality: SubtitleQualityConfig = field(default_factory=SubtitleQualityConfig)

@dataclass
class SubtitleStyleConfigBinding:
    styles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    selected_style: str = "default"
    overrides: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProviderConfig:
    name: str
    options: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GeminiConfig:
    model: str = "gemini-3.5-flash-lite"
    primary_model: str = "gemini-3.5-flash-lite"
    escalation_model: str = "gemini-3.5-flash"
    escalation_enabled: bool = False
    region_detection_model: str = "gemini-3.5-flash-lite"
    region_classification_model: str = "gemini-3.5-flash-lite"
    crop_recognition_model: str = "gemini-3.5-flash-lite"
    verification_model: str = "gemini-3.5-flash-lite"
    region_window_frame_count: int = 8
    region_window_stride: int = 4
    representative_crop_count: int = 3
    max_verification_requests: int = 1
    rpm_cap: int = 60
    tpm_cap: Optional[int] = None
    rpd_cap: Optional[int] = None
    max_concurrency: int = 1
    max_retries: int = 3
    cache_dir: Optional[str] = None
    cache_enabled: bool = True
    cache_namespace: str = "shared"
    validate_models: bool = False
    scheduler: Dict[str, Any] = field(default_factory=dict)
    ocr_strategy: str = "gemini_video_direct"
    direct_video_model: str = "gemini-3.5-flash-lite"
    direct_video_discovery_fps: float = 6
    direct_video_classification_fps: float = 6
    direct_video_segmentation_fps: float = 12
    direct_video_geometry_fps: float = 12
    direct_video_transcription_fps: float = 12
    direct_video_media_resolution: str = "high"
    direct_video_clip_padding_seconds: float = 0.40
    direct_video_upload_mode: str = "files_api"
    direct_video_active_rpm: int = 15
    direct_video_rpm_safety_ratio: float = 0.80
    direct_video_max_in_flight_requests: int = 1
    direct_video_tpm_safety_ratio: float = 0.85
    direct_video_max_requests_per_job: int = 0
    direct_video_max_retry_per_stage: int = 1
    direct_video_max_raw_candidates: int = 200
    direct_video_max_parseable_raw_candidates: int = 200
    direct_video_max_deduplicated_candidates: int = 10
    direct_video_max_accepted_subtitle_tracks: int = 10
    direct_video_max_events_per_track: int = 20
    direct_video_max_total_stage_requests: int = 100
    direct_video_max_counted_input_tokens_per_job: int = 500000
    direct_video_stage_timeout_seconds: float = 360
    direct_video_no_progress_timeout_seconds: float = 300
    direct_video_job_timeout_seconds: float = 600
    direct_video_max_quota_wait_seconds: float = 120
    direct_video_max_quota_retries_per_stage: int = 1
    direct_video_honor_retry_after: bool = True
    direct_video_fail_fast_on_daily_quota: bool = True
    direct_video_min_request_spacing_seconds: float = 1.0
    direct_video_quota_jitter_seconds: float = 0.5
    direct_video_unknown_429_backoff_seconds: float = 5.0
    direct_video_fps_certification_ttl_seconds: float = 86400.0
    direct_video_resume_enabled: bool = True
    direct_video_checkpoint_schema_version: str = "1.0.0"
    direct_video_batch_size: int = 10

    def __post_init__(self) -> None:
        if self.model != "gemini-3.5-flash-lite" and self.primary_model == "gemini-3.5-flash-lite":
            self.primary_model = self.model
        elif self.primary_model != "gemini-3.5-flash-lite":
            self.model = self.primary_model

        if not self.primary_model.strip():
            raise ValueError("Gemini primary model must not be empty.")
        if self.max_retries < 1:
            raise ValueError("Gemini max_retries must be at least 1.")
        if self.rpm_cap < 1 or self.max_concurrency < 1:
            raise ValueError("Gemini scheduler caps must be positive.")
        if self.region_window_frame_count < 2:
            raise ValueError("region_window_frame_count must be at least 2.")
        if not 1 <= self.region_window_stride <= self.region_window_frame_count:
            raise ValueError("region_window_stride must be within the window size.")
        if not 1 <= self.representative_crop_count <= 5:
            raise ValueError("representative_crop_count must be between 1 and 5.")
        if self.max_verification_requests < 0:
            raise ValueError("max_verification_requests must be non-negative.")
        for model in (
            self.region_detection_model,
            self.region_classification_model,
            self.crop_recognition_model,
            self.verification_model,
        ):
            if model not in {"gemini-3.1-flash-lite", "gemini-3.5-flash-lite"}:
                raise ValueError("Phase G.3 OCR stages require a supported Gemini Flash-Lite model.")
        if not self.cache_namespace.strip():
            raise ValueError("Gemini cache_namespace must not be empty.")
        if self.direct_video_model not in {"gemini-3.1-flash-lite", "gemini-3.5-flash-lite"}:
            raise ValueError("G.5 direct video requires a supported Gemini Flash-Lite model.")
        if self.direct_video_active_rpm <= 0:
            raise ValueError("direct_video_active_rpm must be positive.")
        if not 0 < self.direct_video_rpm_safety_ratio <= 1:
            raise ValueError("direct_video_rpm_safety_ratio must be in (0, 1].")
        if self.direct_video_max_in_flight_requests < 1:
            raise ValueError("direct_video_max_in_flight_requests must be positive.")
        if any(fps <= 0 for fps in (
            self.direct_video_discovery_fps, self.direct_video_classification_fps,
            self.direct_video_segmentation_fps, self.direct_video_geometry_fps,
            self.direct_video_transcription_fps,
        )):
            raise ValueError("Direct-video FPS values must be positive.")
        if self.direct_video_max_raw_candidates < 1 or self.direct_video_max_parseable_raw_candidates < 1:
            raise ValueError("direct_video raw candidate caps must be at least 1.")
        if self.direct_video_max_deduplicated_candidates < 1:
            raise ValueError("direct_video_max_deduplicated_candidates must be at least 1.")
        if self.direct_video_max_accepted_subtitle_tracks < 1:
            raise ValueError("direct_video_max_accepted_subtitle_tracks must be at least 1.")
        if self.direct_video_max_events_per_track < 1:
            raise ValueError("direct_video_max_events_per_track must be at least 1.")
        if self.direct_video_max_total_stage_requests < 1:
            raise ValueError("direct_video_max_total_stage_requests must be at least 1.")
        if self.direct_video_max_counted_input_tokens_per_job < 1:
            raise ValueError("direct_video_max_counted_input_tokens_per_job must be at least 1.")
        if self.direct_video_stage_timeout_seconds <= 0:
            raise ValueError("direct_video_stage_timeout_seconds must be positive.")
        if self.direct_video_no_progress_timeout_seconds <= 0:
            raise ValueError("direct_video_no_progress_timeout_seconds must be positive.")
        if self.direct_video_job_timeout_seconds <= 0 or self.direct_video_max_quota_wait_seconds < 0:
            raise ValueError("direct-video time limits must be non-negative/positive as applicable.")
        if self.direct_video_max_quota_retries_per_stage < 0:
            raise ValueError("direct_video_max_quota_retries_per_stage must be non-negative.")
        if min(
            self.direct_video_min_request_spacing_seconds,
            self.direct_video_quota_jitter_seconds,
            self.direct_video_unknown_429_backoff_seconds,
            self.direct_video_fps_certification_ttl_seconds,
        ) < 0:
            raise ValueError("Direct-video quota timing values must be non-negative.")
        if self.direct_video_batch_size < 1:
            raise ValueError("direct_video_batch_size must be at least 1.")
        if not self.direct_video_checkpoint_schema_version.strip():
            raise ValueError("direct_video_checkpoint_schema_version must not be empty.")

@dataclass
class ProviderFallbackConfig:
    ocr_fallback_provider: str = "paddle"
    speech_fallback_provider: str = "faster-whisper"
    ocr_fallback_enabled: bool = True
    speech_fallback_enabled: bool = True

@dataclass
class AppConfig:
    defaults: Dict[str, str] = field(
        default_factory=lambda: {
            "ocr": "gemini_vision",
            "speech": "gemini_audio",
        }
    )
    providers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    subtitle_quality: SubtitleQualityConfig = field(default_factory=SubtitleQualityConfig)
    subtitle_placement: PlacementConfig = field(default_factory=lambda: PlacementConfig(
            padding_px=24,
            preferred_relation="override",
            fallback_alignment=2,
            fallback_margin_v=15,
            min_region_confidence=0.70,
            video_width=1920,
            video_height=1080,
            placement_mode="replace_source_subtitle",
        ))
    styles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    resource_policy: ResourcePolicy = field(default_factory=ResourcePolicy)
    ffmpeg: FFmpegConfig = field(default_factory=FFmpegConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    provider_fallback: ProviderFallbackConfig = field(
        default_factory=ProviderFallbackConfig
    )

    def get_provider_config(self, category: str, override_name: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> ProviderConfig:
        name = override_name or self.defaults.get(category, "mock")
        options = dict(self.providers.get(category, {}).get(name, {}))
        if overrides:
            options.update({k: v for k, v in overrides.items() if v is not None})
        return ProviderConfig(name=name, options=options)
