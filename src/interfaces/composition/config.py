"""Validated provider configuration with deterministic precedence."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.core.errors import ConfigurationError

CODE_DEFAULTS = {"speech": "auto", "translation": "auto", "tts": "auto", "ocr": "auto"}


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConfigurationError(f"Duplicate configuration key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


@dataclass(frozen=True)
class ProviderConfig:
    defaults: Mapping[str, str] = field(default_factory=dict)
    providers: Mapping[str, Mapping[str, Mapping[str, Any]]] = field(default_factory=dict)
    source: str | None = None

    def resolve(self, capability: str, runtime_choice: str = "auto", environ: Mapping[str, str] | None = None) -> str:
        if runtime_choice and runtime_choice != "auto":
            return runtime_choice
        env = environ if environ is not None else os.environ
        choice = env.get(f"MEDIAFLOW_{capability.upper()}_PROVIDER") or self.defaults.get(capability) or CODE_DEFAULTS.get(capability)
        if not choice:
            raise ConfigurationError(f"No default provider is defined for capability '{capability}'.")
        return str(choice)

    def options(self, capability: str, provider: str) -> dict[str, Any]:
        return dict(self.providers.get(capability, {}).get(provider, {}))


def load_provider_config(path: str | Path | None = None) -> ProviderConfig:
    config_path = Path(path) if path else Path(__file__).resolve().parents[3] / "config" / "providers.yaml"
    if not config_path.is_file():
        return ProviderConfig(source=str(config_path))
    try:
        raw = yaml.load(config_path.read_text(encoding="utf-8-sig"), Loader=_UniqueKeyLoader) or {}
    except ConfigurationError:
        raise
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Invalid provider configuration at {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError("Provider configuration root must be a mapping.")
    defaults, providers = raw.get("defaults", {}), raw.get("providers", {})
    if not isinstance(defaults, dict) or not isinstance(providers, dict):
        raise ConfigurationError("Provider configuration 'defaults' and 'providers' must be mappings.")
    for capability, choice in defaults.items():
        if not isinstance(capability, str) or not isinstance(choice, str) or not choice.strip():
            raise ConfigurationError("Provider defaults must use non-empty string keys and values.")
        configured = providers.get(capability, {})
        if choice not in {"auto", "mock"} and configured and choice not in configured:
            raise ConfigurationError(f"Default provider '{choice}' is not configured under providers.{capability}.")
    return ProviderConfig(dict(defaults), dict(providers), str(config_path))
