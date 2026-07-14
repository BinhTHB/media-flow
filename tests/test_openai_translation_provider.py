import os
import pytest
from src.domain.entities.segment import Segment
from src.infrastructure.providers.translation.openai_translation_provider import OpenAITranslationProvider
from src.shared.exceptions import ProviderError

def test_openai_api_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OpenAITranslationProvider()
    segments = [Segment(id="1", start=0, end=1, text="Hello", source="speech")]
    with pytest.raises(ProviderError, match="OPENAI_API_KEY environment variable is not set"):
        provider.translate(segments, "vi")

def test_openai_translation_mock(monkeypatch):
    class MockChatCompletions:
        def create(self, *args, **kwargs):
            class MockChoice:
                class MockMessage:
                    content = "0: Xin chào"
                message = MockMessage()
            class MockResponse:
                choices = [MockChoice()]
            return MockResponse()

    class MockChat:
        completions = MockChatCompletions()

    class MockOpenAI:
        def __init__(self, api_key):
            self.chat = MockChat()

    monkeypatch.setenv("OPENAI_API_KEY", "dummy_key")
    # Mock the openai package structure
    import sys
    from types import ModuleType
    mock_openai_module = ModuleType("openai")
    mock_openai_module.OpenAI = MockOpenAI
    sys.modules["openai"] = mock_openai_module

    provider = OpenAITranslationProvider()
    segments = [Segment(id="1", start=0, end=1, text="Hello", source="speech")]
    result = provider.translate(segments, "vi")
    assert result[0].translated_text == "Xin chào"
