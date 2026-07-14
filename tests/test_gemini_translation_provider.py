import json
import pytest

from src.domain.entities.segment import Segment
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
        "src.infrastructure.providers.translation.gemini_translation_provider.request.urlopen",
        lambda *args, **kwargs: FakeResponse(),
    )

    provider = GeminiTranslationProvider()
    segments = [Segment(id="1", start=0, end=1, text="Hello", source="speech")]
    result = provider.translate(segments, "vi")
    assert result[0].translated_text == "Xin chào"
