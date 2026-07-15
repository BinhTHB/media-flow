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

        defaults = providers_raw.get("defaults", {})
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
        gemini = GeminiConfig(
            model=os.environ.get("GEMINI_MODEL", gemini_raw.get("model", GeminiConfig.model)),
            max_retries=self._env_int(
                "GEMINI_MAX_RETRIES",
                gemini_overrides.get("max_retries", gemini_raw.get("max_retries", 3)),
            ),
            cache_dir=os.environ.get(
                "GEMINI_CACHE_DIR",
                gemini_overrides.get("cache_dir", gemini_raw.get("cache_dir")),
            ),
            cache_namespace=os.environ.get(
                "GEMINI_CACHE_NAMESPACE",
                gemini_overrides.get(
                    "cache_namespace", gemini_raw.get("cache_namespace", "shared")
                ),
            ),
            cache_enabled=self._env_bool(
                "GEMINI_CACHE_ENABLED",
                gemini_overrides.get(
                    "cache_enabled", gemini_raw.get("cache_enabled", True)
                ),
            ),
            scheduler=dict(
                gemini_overrides.get(
                    "scheduler", gemini_raw.get("scheduler", {})
                )
            ),
            file_manager_enabled=self._env_bool(
                "GEMINI_FILE_MANAGER_ENABLED",
                gemini_overrides.get(
                    "file_manager_enabled",
                    gemini_raw.get("file_manager_enabled", False),
                ),
            ),
        )

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
