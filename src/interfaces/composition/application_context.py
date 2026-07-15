from dataclasses import dataclass
from typing import Optional

from src.infrastructure.gemini.runtime import GeminiRuntime
from src.interfaces.config.app_config import AppConfig
from src.infrastructure.registry.provider_registry import ProviderRegistry
from src.infrastructure.factory.provider_factory import ProviderFactory
from src.infrastructure.artifacts.artifact_manager import ArtifactManager

@dataclass
class ApplicationContext:
    config: AppConfig
    provider_registry: ProviderRegistry
    provider_factory: ProviderFactory
    artifact_manager: ArtifactManager
    gemini_runtime: Optional[GeminiRuntime] = None
