import os
from typing import Any, Dict, Optional
import yaml

from src.domain.entities.placement_config import PlacementConfig
from src.domain.entities.resource_policy import FFmpegConfig, ResourcePolicy
from src.domain.entities.subtitle_quality_config import SubtitleQualityConfig
from src.domain.entities.subtitle_style import SubtitleStyle
from src.interfaces.config.app_config import (
    AppConfig,
    GeminiConfig,
    OutputConfig,
    ProviderFallbackConfig,
    RuntimeConfig,
)


class ConfigLoader:
    def __init__(
        self,
        providers_path: Optional[str] = None,
        subtitle_styles_path: Optional[str] = None,
    ):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        self.providers_path = providers_path or os.path.join(base_dir, "config", "providers.yaml")
        self.subtitle_styles_path = subtitle_styles_path or os.path.join(base_dir, "config", "subtitle_styles.yaml")
        self.secrets_path = os.path.join(base_dir, "config", "config.yml")
        self.gemini_api_key: Optional[str] = None

    def load(self, overrides: Optional[Dict[str, Any]] = None) -> AppConfig:
        providers_raw = self._load_yaml(self.providers_path)
        styles_raw = self._load_yaml(self.subtitle_styles_path)
        secrets_raw = self._load_yaml(self.secrets_path)

        defaults = {
            "ocr": "gemini_vision",
            "speech": "gemini_audio",
            **providers_raw.get("defaults", {}),
        }
        if os.path.abspath(self.providers_path) == os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../config/providers.yaml")
        ):
            defaults.update({"ocr": "gemini_vision", "speech": "gemini_audio"})
        providers = {
            category: {
                name: {
                    key: value
                    for key, value in options.items()
                    if not (category == "translation" and name == "gemini" and key == "api_key")
                }
                for name, options in configured.items()
            }
            for category, configured in providers_raw.get("providers", {}).items()
        }
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY") or secrets_raw.get(
            "gemini", {}
        ).get("api_key")
        gemini_raw = providers.get("translation", {}).get("gemini", {})
        quality_raw = providers_raw.get("subtitle_quality", {})
        placement_raw = providers_raw.get("subtitle_placement", {})
        styles = styles_raw.get("styles", {})
        self._validate_subtitle_config_ownership(quality_raw, styles)

        overrides = overrides or {}
        quality_overrides = overrides.get("subtitle_quality", {})
        placement_overrides = overrides.get("subtitle_placement", {})
        runtime_overrides = overrides.get("runtime", {})
        output_overrides = overrides.get("output", {})
        resource_overrides = overrides.get("resource_policy", {})
        ffmpeg_overrides = overrides.get("ffmpeg", {})
        gemini_overrides = overrides.get("gemini", {})

        default_quality = SubtitleQualityConfig()
        quality_values = {
            key: quality_raw.get(key, getattr(default_quality, key))
            for key in SubtitleQualityConfig.__dataclass_fields__
        }
        quality_values.update(quality_overrides)
        quality = SubtitleQualityConfig(**quality_values)

        default_placement = PlacementConfig(
            padding_px=24,
            preferred_relation="below_source",
            fallback_alignment=2,
            fallback_margin_v=15,
            min_region_confidence=0.70,
            video_width=1920,
            video_height=1080,
        )
        placement_values = {
            key: placement_raw.get(key, getattr(default_placement, key))
            for key in PlacementConfig.__dataclass_fields__
        }
        placement_values.update(placement_overrides)
        placement = PlacementConfig(**placement_values)
        primary_model = os.environ.get(
            "GEMINI_PRIMARY_MODEL",
            os.environ.get("GEMINI_MODEL", gemini_overrides.get("model", gemini_raw.get("model", GeminiConfig.model))),
        )
        gemini = GeminiConfig(
            model=primary_model,
            escalation_model=os.environ.get(
                "GEMINI_ESCALATION_MODEL",
                gemini_overrides.get("escalation_model", gemini_raw.get("escalation_model", GeminiConfig.escalation_model)),
            ),
            escalation_enabled=self._env_bool(
                "GEMINI_ESCALATION_ENABLED",
                gemini_overrides.get("escalation_enabled", gemini_raw.get("escalation_enabled", False)),
            ),
            rpm_cap=self._env_int("GEMINI_RPM_CAP", gemini_overrides.get("rpm_cap", gemini_raw.get("rpm_cap", 60))),
            tpm_cap=self._env_optional_int("GEMINI_TPM_CAP", gemini_overrides.get("tpm_cap", gemini_raw.get("tpm_cap"))),
            rpd_cap=self._env_optional_int("GEMINI_RPD_CAP", gemini_overrides.get("rpd_cap", gemini_raw.get("rpd_cap"))),
            max_concurrency=self._env_int(
                "GEMINI_MAX_CONCURRENCY",
                gemini_overrides.get("max_concurrency", gemini_raw.get("max_concurrency", 1)),
            ),
            max_retries=self._env_int(
                "GEMINI_MAX_RETRIES",
                gemini_overrides.get("max_retries", gemini_raw.get("max_retries", 3)),
            ),
            cache_dir=os.environ.get(
                "GEMINI_CACHE_DIR",
                gemini_overrides.get("cache_dir", gemini_raw.get("cache_dir")),
            ),
            cache_enabled=self._env_bool(
                "GEMINI_CACHE_ENABLED",
                gemini_overrides.get(
                    "cache_enabled", gemini_raw.get("cache_enabled", True)
                ),
            ),
            validate_models=self._env_bool(
                "GEMINI_VALIDATE_MODELS",
                gemini_overrides.get("validate_models", gemini_raw.get("validate_models", False)),
            ),
            ocr_strategy=os.environ.get(
                "GEMINI_OCR_STRATEGY",
                gemini_overrides.get("ocr_strategy", gemini_raw.get("ocr_strategy", "gemini_video_direct")),
            ),
            direct_video_model=os.environ.get(
                "GEMINI_DIRECT_VIDEO_MODEL",
                gemini_overrides.get("direct_video_model", gemini_raw.get("direct_video_model", GeminiConfig.direct_video_model)),
            ),
            direct_video_discovery_fps=self._env_float(
                "GEMINI_DIRECT_VIDEO_DISCOVERY_FPS",
                gemini_overrides.get("direct_video_discovery_fps", gemini_raw.get("direct_video_discovery_fps", 6)),
            ),
            direct_video_classification_fps=self._env_float(
                "GEMINI_DIRECT_VIDEO_CLASSIFICATION_FPS",
                gemini_overrides.get("direct_video_classification_fps", gemini_raw.get("direct_video_classification_fps", 6)),
            ),
            direct_video_segmentation_fps=self._env_float(
                "GEMINI_DIRECT_VIDEO_SEGMENTATION_FPS",
                gemini_overrides.get("direct_video_segmentation_fps", gemini_raw.get("direct_video_segmentation_fps", 12)),
            ),
            direct_video_geometry_fps=self._env_float(
                "GEMINI_DIRECT_VIDEO_GEOMETRY_FPS",
                gemini_overrides.get("direct_video_geometry_fps", gemini_raw.get("direct_video_geometry_fps", 12)),
            ),
            direct_video_transcription_fps=self._env_float(
                "GEMINI_DIRECT_VIDEO_TRANSCRIPTION_FPS",
                gemini_overrides.get("direct_video_transcription_fps", gemini_raw.get("direct_video_transcription_fps", 12)),
            ),
            direct_video_media_resolution=os.environ.get(
                "GEMINI_DIRECT_VIDEO_MEDIA_RESOLUTION",
                gemini_overrides.get("direct_video_media_resolution", gemini_raw.get("direct_video_media_resolution", "high")),
            ),
            direct_video_clip_padding_seconds=self._env_float(
                "GEMINI_DIRECT_VIDEO_CLIP_PADDING_SECONDS",
                gemini_overrides.get("direct_video_clip_padding_seconds", gemini_raw.get("direct_video_clip_padding_seconds", 0.40)),
            ),
            direct_video_upload_mode=os.environ.get(
                "GEMINI_DIRECT_VIDEO_UPLOAD_MODE",
                gemini_overrides.get("direct_video_upload_mode", gemini_raw.get("direct_video_upload_mode", "files_api")),
            ),
            direct_video_tpm_safety_ratio=self._env_float(
                "GEMINI_DIRECT_VIDEO_TPM_SAFETY_RATIO",
                gemini_overrides.get("direct_video_tpm_safety_ratio", gemini_raw.get("direct_video_tpm_safety_ratio", 0.85)),
            ),
            direct_video_max_requests_per_job=self._env_int(
                "GEMINI_DIRECT_VIDEO_MAX_REQUESTS_PER_JOB",
                gemini_overrides.get("direct_video_max_requests_per_job", gemini_raw.get("direct_video_max_requests_per_job", 0)),
            ),
            direct_video_max_retry_per_stage=self._env_int(
                "GEMINI_DIRECT_VIDEO_MAX_RETRY_PER_STAGE",
                gemini_overrides.get("direct_video_max_retry_per_stage", gemini_raw.get("direct_video_max_retry_per_stage", 1)),
            ),
        )
        fallback_raw = providers_raw.get("provider_fallback", {})

        return AppConfig(
            defaults=defaults,
            providers=providers,
            subtitle_quality=quality,
            subtitle_placement=placement,
            styles=styles,
            runtime=RuntimeConfig(**runtime_overrides),
            output=OutputConfig(**output_overrides),
            resource_policy=ResourcePolicy(
                **{**providers_raw.get("resource_policy", {}), **resource_overrides}
            ),
            ffmpeg=FFmpegConfig(
                **{**providers_raw.get("ffmpeg", {}), **ffmpeg_overrides}
            ),
            gemini=gemini,
            provider_fallback=ProviderFallbackConfig(
                ocr_fallback_provider=fallback_raw.get(
                    "ocr_fallback_provider", "paddle"
                ),
                speech_fallback_provider=fallback_raw.get(
                    "speech_fallback_provider", "faster-whisper"
                ),
                ocr_fallback_enabled=bool(
                    fallback_raw.get("ocr_fallback_enabled", True)
                ),
                speech_fallback_enabled=bool(
                    fallback_raw.get("speech_fallback_enabled", True)
                ),
            ),
        )

    @staticmethod
    def _load_yaml(path: str) -> Dict[str, Any]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    @staticmethod
    def _env_int(name: str, default: Any) -> int:
        value = os.environ.get(name)
        try:
            return int(value) if value is not None else int(default)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer.") from exc

    @staticmethod
    def _env_int_optional(name: str, default: Any) -> Optional[int]:
        value = os.environ.get(name)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be an integer.") from exc
        if isinstance(default, int):
            return default
        return None

    @staticmethod
    def _env_float(name: str, default: Any) -> float:
        value = os.environ.get(name)
        try:
            return float(value) if value is not None else float(default)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a number.") from exc

    @staticmethod
    def _env_bool(name: str, default: Any) -> bool:
        value = os.environ.get(name)
        if value is None:
            return bool(default)
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{name} must be a boolean.")

    @staticmethod
    def _env_optional_int(name: str, default: Any) -> Optional[int]:
        value = os.environ.get(name)
        if value is None:
            return None if default is None else ConfigLoader._env_int(name, default)
        return ConfigLoader._env_int(name, value)

    @staticmethod
    def _validate_subtitle_config_ownership(
        quality: Dict[str, Any], styles: Dict[str, Dict[str, Any]]
    ) -> None:
        quality_keys = set(SubtitleQualityConfig.__dataclass_fields__)
        style_keys = set(SubtitleStyle.__dataclass_fields__)
        visual_keys_in_quality = set(quality) & style_keys
        if visual_keys_in_quality:
            raise ValueError(
                "subtitle_quality cannot contain visual style keys: "
                f"{sorted(visual_keys_in_quality)}"
            )
        unknown_quality_keys = set(quality) - quality_keys
        if unknown_quality_keys:
            raise ValueError(
                "subtitle_quality contains unsupported keys: "
                f"{sorted(unknown_quality_keys)}"
            )
        for name, style in styles.items():
            quality_keys_in_style = set(style) & quality_keys
            if quality_keys_in_style:
                raise ValueError(
                    f"subtitle style '{name}' cannot contain quality keys: "
                    f"{sorted(quality_keys_in_style)}"
                )
            unknown_keys = set(style) - style_keys
            if unknown_keys:
                raise ValueError(
                    f"subtitle style '{name}' contains unsupported keys: "
                    f"{sorted(unknown_keys)}"
                )
