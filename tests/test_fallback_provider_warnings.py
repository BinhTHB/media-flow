import logging
from src.infrastructure.providers.provider_fallback import FallbackOCRProvider
from src.infrastructure.gemini.exceptions import GeminiRateLimitError
from src.domain.entities.segment import Segment
from src.domain.entities.frame import Frame

class _PrimaryOCR:
    validation = [{"attempt": 1}]
    def detect_text(self, frames):
        raise GeminiRateLimitError("quota exhausted")

class _FallbackOCR:
    def detect_text(self, frames):
        return [Segment("ocr", 0, 1, "ok", source="ocr")]

def test_fallback_provider_logs_warning_on_failure(caplog):
    provider = FallbackOCRProvider(_PrimaryOCR(), _FallbackOCR(), "gemini", "paddle")
    with caplog.at_level(logging.WARNING):
        provider.detect_text([Frame(timestamp=0, image_path="unused")])
    assert "LOUD-WARN: Primary provider 'gemini' failed" in caplog.text
    assert "Falling back to 'paddle'" in caplog.text
