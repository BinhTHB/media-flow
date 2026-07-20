import logging
from pathlib import Path

import pytest

from src.interfaces.config.config_loader import ConfigLoader
from src.infrastructure.gemini.runtime import GeminiRuntime, RuntimeConfig


def test_providers_yaml_sets_gemini_cache_dir():
    # RED: providers.yaml currently has cache_enabled but no cache_dir,
    # so ResponseCache is silently disabled. Fix sets cache_dir to a
    # repo-internal path that .gitignore covers.
    cfg = ConfigLoader().load()
    assert cfg.gemini.cache_dir, "gemini.cache_dir must be set so the disk cache is enabled"
    # Must be relative to repo root and gitignored (under .cache/)
    assert "cache" in cfg.gemini.cache_dir.replace("\\", "/")


def test_runtime_logs_effective_cache_state(caplog):
    with caplog.at_level(logging.INFO, logger="src.infrastructure.gemini.runtime"):
        runtime = GeminiRuntime.create(
            RuntimeConfig(api_key="fake", cache_dir=".cache/gemini")
        )
    # Effective cache state must be logged loudly once at startup.
    assert any(
        "cache" in rec.message.lower() and ("enabled" in rec.message.lower() or "disabled" in rec.message.lower())
        for rec in caplog.records
    ), "GeminiRuntime must log effective cache state (enabled/disabled + dir) at startup"
