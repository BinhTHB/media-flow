from typing import Any, Optional
from src.infrastructure.gemini.runtime import GeminiRuntime, RuntimeConfig
from src.infrastructure.registry.provider_registry import ProviderRegistry

class ProviderFactory:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        registry: ProviderRegistry | None = None,
        gemini_config: Any = None,
        gemini_api_key: Optional[str] = None,
    ):
        self.config = config or {}
        self.registry = registry or ProviderRegistry()
        self.gemini_config = gemini_config
        self.gemini_api_key = gemini_api_key
        self._gemini_runtime: Optional[GeminiRuntime] = None

    def default_name(self, category: str) -> str:
        return self.config.get("defaults", {}).get(category, "mock")

    def provider_options(self, category: str, name: str) -> dict[str, Any]:
        return dict(self.config.get("providers", {}).get(category, {}).get(name, {}))

    def _merged_options(self, category: str, name: str | None, overrides: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
        provider_name = name or self.default_name(category)
        options = self.provider_options(category, provider_name)
        if overrides:
            options.update({k: v for k, v in overrides.items() if v is not None})
        return provider_name, options

    def create_provider(self, category: str, name: str | None = None, overrides: dict[str, Any] | None = None):
        provider_name, options = self._merged_options(category, name, overrides)
        return self.registry.create(category, provider_name, options)

    def create_video_provider(self, name: str | None = None, overrides: dict[str, Any] | None = None):
        return self.create_provider("video", name, overrides)

    def create_speech_provider(self, name: str | None = None, overrides: dict[str, Any] | None = None):
        return self.create_provider("speech", name, overrides)

    def create_translation_provider(self, name: str | None = None, overrides: dict[str, Any] | None = None):
        provider_name, options = self._merged_options("translation", name, overrides)
        if provider_name == "gemini":
            options["runtime"] = self._get_gemini_runtime(options)
        return self.registry.create("translation", provider_name, options)

    def _get_gemini_runtime(self, options: dict[str, Any]) -> GeminiRuntime:
        if self._gemini_runtime is None:
            config = self.gemini_config
            self._gemini_runtime = GeminiRuntime.create(
                RuntimeConfig(
                    api_key=self.gemini_api_key,
                    default_model=getattr(config, "model", options.get("model", "gemini-3.1-flash-lite")),
                    cache_dir=getattr(config, "cache_dir", options.get("cache_dir")),
                    cache_namespace=getattr(config, "cache_namespace", options.get("cache_namespace", "shared")),
                    cache_enabled=getattr(config, "cache_enabled", options.get("cache_enabled", True)),
                    scheduler=dict(getattr(config, "scheduler", options.get("scheduler", {}))),
                    file_manager_enabled=getattr(
                        config,
                        "file_manager_enabled",
                        options.get("file_manager_enabled", False),
                    ),
                )
            )
        return self._gemini_runtime

    @property
    def gemini_runtime(self) -> Optional[GeminiRuntime]:
        return self._gemini_runtime

    def create_subtitle_provider(self, name: str | None = None, overrides: dict[str, Any] | None = None):
        return self.create_provider("subtitle", name, overrides)

    def create_subtitle_region_detector_provider(self, name: str | None = None, overrides: dict[str, Any] | None = None):
        return self.create_provider("subtitle_region_detector", name, overrides)

    def create_subtitle_placement_debug_provider(self, name: str | None = None, overrides: dict[str, Any] | None = None):
        return self.create_provider("subtitle_placement_debug", name, overrides)

    def create_ocr_provider(self, name: str | None = None, overrides: dict[str, Any] | None = None):
        return self.create_provider("ocr", name, overrides)

    def create_storage_provider(self, name: str | None = None, overrides: dict[str, Any] | None = None):
        return self.create_provider("storage", name, overrides)

    def create_frame_extraction_provider(self, name: str | None = None, overrides: dict[str, Any] | None = None):
        return self.create_provider("frame_extraction", name, overrides)

    def create_tts_provider(self, name: str | None = None, overrides: dict[str, Any] | None = None):
        return self.create_provider("tts", name or "edge-tts", overrides)

    def create_dubbing_provider(self, name: str | None = None, overrides: dict[str, Any] | None = None):
        return self.create_provider("dubbing", name or "ffmpeg", overrides)
