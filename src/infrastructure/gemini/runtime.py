from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.infrastructure.gemini.client import GeminiClient
from src.infrastructure.gemini.contracts import DEFAULT_MODEL, MODE_ROLE_PRIMARY
from src.infrastructure.gemini.file_manager import FileManager
from src.infrastructure.gemini.request_scheduler import RequestScheduler
from src.infrastructure.gemini.response_cache import ResponseCache
from src.infrastructure.gemini.usage_recorder import UsageRecorder, UsageRecord


logger = logging.getLogger(__name__)


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


def _create_ledger_callback(ledger_path: str, job_id: str):
    import os, json, time, threading
    _ledger_path = ledger_path
    _job_id = job_id
    _lock = threading.Lock()
    def callback(record: UsageRecord):
        if record.job_id is None:
            return
        try:
            os.makedirs(os.path.dirname(_ledger_path), exist_ok=True)
            entry = {
                "timestamp": time.time(),
                "job_id": record.job_id,
                "purpose": record.source_kind or "unknown",
                "model": record.model,
                "live_or_cache": "cache" if record.cache_hit else "live",
                "request_id_or_hash": record.request_id or "",
                "status": "success" if record.success else "error",
                "tokens": record.total_tokens,
                "error_msg": record.error_msg or "",
            }
            with _lock:
                with open(_ledger_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("LOUD-WARN: Failed to write to request ledger at %s: %s", _ledger_path, str(e))
    return callback


class GeminiRuntime:
    """Application-scoped runtime that owns shared Gemini infrastructure components.

    All Gemini clients created through a runtime share the same scheduler, cache
    and usage recorder, so OCR/ASR/translation requests are throttled and accounted
    together (per the shared-project-quota strategy).
    """

    def __init__(self, config: Optional[RuntimeConfig] = None, job_id: str = "default") -> None:
        self._config = config or RuntimeConfig()
        self._job_id = job_id
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
        logger.warning(
            "Gemini response cache %s (dir=%s)",
            "enabled" if self._cache.enabled else "disabled",
            self._cache.path or self._config.cache_dir or "unset",
        )
        self._usage = UsageRecorder(
            usage_callback=_create_ledger_callback(self._ledger_path_for(job_id), job_id)
        )
        self._files = FileManager(enabled=self._config.file_manager_enabled)
        self._clients: Dict[str, GeminiClient] = {}

    @classmethod
    def create(cls, config: Optional[RuntimeConfig] = None, job_id: str = "default") -> "GeminiRuntime":
        return cls(config, job_id=job_id)

    def set_job_id(self, job_id: str) -> None:
        """Set per-job identifier and recreate ledger callback to write to the per-job ledger file."""
        self._job_id = job_id
        self._usage = UsageRecorder(
            usage_callback=_create_ledger_callback(self._ledger_path_for(job_id), job_id)
        )
        self._clients.clear()

    def _ledger_path_for(self, job_id: str) -> str:
        """Derive ledger file path from RuntimeConfig.ledger_path.

        Returns the configured path as-is if the job_id is already part of the filename.
        Otherwise, appends the job_id before the extension.
        C?m: no literal 'round19'/'round20'/'round21' in src/.
        """
        cfg_path = self._config.ledger_path
        if not cfg_path or cfg_path == "debug/gemini_request_ledger.json":
            # Default fallback
            base = os.path.join("debug")
            os.makedirs(base, exist_ok=True)
            return os.path.join(base, f"ledger_{job_id}.jsonl")

        # If job_id is already in the filename, return as-is
        base_name = os.path.basename(cfg_path)
        if job_id.lower() in base_name.lower():
            dir_path = os.path.dirname(cfg_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            return cfg_path

        # Otherwise, append job_id before extension
        dir_path = os.path.dirname(cfg_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        name, ext = os.path.splitext(base_name)
        fname = f"{name}_{job_id}{ext}"
        return os.path.join(dir_path, fname) if dir_path else fname

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
                job_id=self._job_id,
            )
            self._clients[cache_key] = client
            if self._files.enabled and not self._files.supports_remote_upload:
                self._files._client = client
        return client
