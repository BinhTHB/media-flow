import json
import urllib.request
import pytest

from src.domain.entities.segment import Segment
from src.infrastructure.factory.provider_factory import ProviderFactory
from src.infrastructure.gemini.runtime import GeminiRuntime
from src.infrastructure.providers.translation.gemini_translation_provider import GeminiTranslationProvider
from src.shared.exceptions import ProviderError


def test_gemini_api_key_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    provider = GeminiTranslationProvider()
    segments = [Segment(id="1", start=0, end=1, text="Hello", source="speech")]
    with pytest.raises(ProviderError, match="GEMINI_API_KEY environment variable is not set"):
        provider.translate(segments, "vi")


def test_gemini_translation_response(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self):
            return json.dumps({
                "candidates": [{"content": {"parts": [{"text": "0: Xin chào"}]}}]
            }).encode("utf-8")

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(),
    )

    provider = GeminiTranslationProvider()
    segments = [Segment(id="1", start=0, end=1, text="Hello", source="speech")]
    result = provider.translate(segments, "vi")
    assert result[0].translated_text == "Xin chào"

def test_gemini_factory_injects_one_runtime():
    factory = ProviderFactory(
        config={
            "defaults": {"translation": "gemini"},
            "providers": {
                "translation": {
                    "gemini": {
                        "model": "gemini-3.5-flash-lite",
                        "max_retries": 5,
                    }
                }
            },
        },
        gemini_api_key="test-key",
    )

    first = factory.create_translation_provider()
    second = factory.create_translation_provider()

    assert isinstance(first._runtime, GeminiRuntime)
    assert first._runtime is second._runtime


def test_cli_parser_accepts_gemini_translation_provider():
    import argparse
    from src.interfaces.cli.demo import build_parser
    from src.interfaces.composition.application_builder import ApplicationBuilder

    args = build_parser().parse_args(["dummy.mp4", "--translation-provider", "gemini"])
    assert args.translation_provider == "gemini"

    builder = ApplicationBuilder.from_cli_args(args)
    assert builder.translation_provider == "gemini"
