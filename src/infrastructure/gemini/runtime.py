from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.infrastructure.gemini.client import GeminiClient
from src.infrastructure.gemini.contracts import DEFAULT_MODEL, MODE_ROLE_PRIMARY
from src.infrastructure.gemini.file_manager import FileManager
from src.infrastructure.gemini.request_scheduler import RequestScheduler
from src.infrastructure.gemini.response_cache import ResponseCache
from src.infrastructure.gemini.usage_recorder import UsageRecorder


@dataclass
class RuntimeConfig:
    api_key: Optional[str] = None
    default_model: str = DEFAULT_MODEL
    cache_dir: Optional[str] = None
    cache_enabled: bool = True
    cache_namespace: str = "shared"
    scheduler: Dict[str, Any] = field(default_factory=dict)
    file_manager_enabled: bool = False
    rpm_cap: int = 60
    tpm_cap: Optional[int] = None
    rpd_cap: Optional[int] = None
    max_concurrency: int = 1
    max_retries: int = 3
    validate_models: bool = False
    ledger_path: Optional[str] = "debug/gemini_request_ledger.json"
    min_request_spacing_seconds: float = 0.0


class GeminiRuntime:
    """Application-scoped runtime that owns shared Gemini infrastructure components.

    All Gemini clients created through a runtime share the same scheduler, cache
    and usage recorder, so OCR/ASR/translation requests are throttled and accounted
    together (per the shared-project-quota strategy).
    """

    def __init__(self, config: Optional[RuntimeConfig] = None) -> None:
        self._config = config or RuntimeConfig()
        scheduler_options = dict(self._config.scheduler)
        scheduler_options.update(
            {
                "rpm_cap": self._config.rpm_cap,
                "tpm_cap": self._config.tpm_cap,
                "rpd_cap": self._config.rpd_cap,
                "max_concurrency": self._config.max_concurrency,
                "max_retries": self._config.max_retries,
                "ledger_path": self._config.ledger_path,
                "min_request_spacing_seconds": self._config.min_request_spacing_seconds,
            }
        )
        self._scheduler = RequestScheduler(**scheduler_options)
        self._cache = ResponseCache(
            cache_dir=self._config.cache_dir,
            namespace=self._config.cache_namespace,
            enabled=self._config.cache_enabled,
        )
        self._usage = UsageRecorder()
        self._files = FileManager(enabled=self._config.file_manager_enabled)
        self._clients: Dict[str, GeminiClient] = {}

    @classmethod
    def create(cls, config: Optional[RuntimeConfig] = None) -> "GeminiRuntime":
        return cls(config)

    @property
    def scheduler(self) -> RequestScheduler:
        return self._scheduler

    @property
    def cache(self) -> ResponseCache:
        return self._cache

    @property
    def usage_recorder(self) -> UsageRecorder:
        return self._usage

    @property
    def file_manager(self) -> FileManager:
        return self._files

    def get_client(self, model: Optional[str] = None, *, role: str = MODE_ROLE_PRIMARY,
                   api_key: Optional[str] = None) -> GeminiClient:
        model = model or self._config.default_model
        cache_key = f"{role}:{model}"
        client = self._clients.get(cache_key)
        if client is None:
            client = GeminiClient(
                model=model,
                api_key=api_key or self._config.api_key,
                scheduler=self._scheduler,
                cache=self._cache,
                usage_recorder=self._usage,
                file_manager=self._files,
                role=role,
                validate_models=self._config.validate_models,
            )
            self._clients[cache_key] = client
            if self._files.enabled and not self._files.supports_remote_upload:
                self._files._client = client
        return client
