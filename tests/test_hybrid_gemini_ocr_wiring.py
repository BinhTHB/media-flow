import os
import sys
import types
import pytest

from src.infrastructure.factory.provider_factory import ProviderFactory
from src.infrastructure.registry.provider_registry import ProviderRegistry
from src.infrastructure.providers.ocr.hybrid_gemini_ocr_provider import HybridGeminiOCRProvider


class _FakeClient:
    def generate_content(self, *args, **kwargs):
        raise AssertionError("should not be called in wiring test")


class _FakeRuntime:
    def get_client(self, model=None, **kwargs):
        return _FakeClient()


def test_factory_wires_hybrid_gemini_ocr():
    registry = ProviderRegistry()
    factory = ProviderFactory(registry=registry, gemini_api_key="dummy")
    provider = factory.create_ocr_provider(
        "hybrid_gemini",
        overrides={"gemini_verify": True, "weights": "best.pt", "device": "cpu"},
    )
    assert isinstance(provider, HybridGeminiOCRProvider)
    assert provider._semantic_verifier is not None
    assert provider._proposal is not None
    assert provider._min_confidence == 0.40


def test_registry_has_hybrid_gemini():
    registry = ProviderRegistry()
    assert "hybrid_gemini" in registry.names("ocr")
