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
    model: str = "gemini-3.1-flash-lite"
    primary_model: str = "gemini-3.1-flash-lite"
    escalation_model: str = "gemini-3.5-flash"
    escalation_enabled: bool = False
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

    def __post_init__(self) -> None:
        if self.model != "gemini-3.1-flash-lite" and self.primary_model == "gemini-3.1-flash-lite":
            self.primary_model = self.model
        elif self.primary_model != "gemini-3.1-flash-lite":
            self.model = self.primary_model

        if not self.primary_model.strip():
            raise ValueError("Gemini primary model must not be empty.")
        if self.max_retries < 1:
            raise ValueError("Gemini max_retries must be at least 1.")
        if self.rpm_cap < 1 or self.max_concurrency < 1:
            raise ValueError("Gemini scheduler caps must be positive.")
        if not self.cache_namespace.strip():
            raise ValueError("Gemini cache_namespace must not be empty.")

@dataclass
class AppConfig:
    defaults: Dict[str, str] = field(default_factory=dict)
    providers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    subtitle_quality: SubtitleQualityConfig = field(default_factory=SubtitleQualityConfig)
    subtitle_placement: PlacementConfig = field(default_factory=lambda: PlacementConfig(
        padding_px=24,
        preferred_relation="below_source",
        fallback_alignment=2,
        fallback_margin_v=15,
        min_region_confidence=0.70,
        video_width=1920,
        video_height=1080,
    ))
    styles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    resource_policy: ResourcePolicy = field(default_factory=ResourcePolicy)
    ffmpeg: FFmpegConfig = field(default_factory=FFmpegConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)

    def get_provider_config(self, category: str, override_name: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> ProviderConfig:
        name = override_name or self.defaults.get(category, "mock")
        options = dict(self.providers.get(category, {}).get(name, {}))
        if overrides:
            options.update({k: v for k, v in overrides.items() if v is not None})
        return ProviderConfig(name=name, options=options)
