from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

from src.domain.entities.subtitle_region import SubtitleRegion
from src.application.ports.ocr_provider import OCRProvider
from src.application.ports.speech_provider import SpeechProvider
from src.domain.entities.frame import Frame
from src.domain.entities.segment import Segment
from src.infrastructure.gemini.exceptions import (
    GeminiAPIError,
    GeminiModelValidationError,
    GeminiNetworkError,
    GeminiRateLimitError,
    GeminiValidationError,
)

logger = logging.getLogger(__name__)


def _fallback_trigger(error: Exception) -> str | None:
    message = str(error).lower()
    if any(token in message for token in ("corrupt wav", "pcm audio", "could not read", "no such file")):
        return None
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, GeminiRateLimitError):
            return "quota_exhausted"
        if isinstance(current, GeminiNetworkError):
            return "timeout_or_transient_network"
        if isinstance(current, GeminiModelValidationError):
            return "model_unavailable"
        if isinstance(current, GeminiValidationError):
            if any(token in str(current).lower() for token in ("wav", "pcm", "audio")):
                return None
            return "structured_response_invalid"
        if isinstance(current, GeminiAPIError) and current.status_code and current.status_code >= 500:
            return "transient_server_failure"
        current = current.__cause__
    return None


class _FallbackProvider:
    def __init__(self, primary: Any, fallback: Any, primary_name: str, fallback_name: str):
        self.primary = primary
        self.fallback = fallback
        self.primary_name = primary_name
        self.fallback_name = fallback_name
        self.provenance: Dict[str, Any] = {}

    def _run(self, operation: Callable[[], List[Segment]]) -> List[Segment]:
        try:
            result = operation()
            self.provenance = {
                "configured_provider": self.primary_name,
                "effective_provider": self.primary_name,
                "fallback_used": False,
            }
            return result
        except Exception as error:
            trigger = _fallback_trigger(error)
            if trigger is None:
                raise
            logger.warning(
                "LOUD-WARN: Primary provider '%s' failed (trigger=%s, error=%s). "
                "Falling back to '%s'.",
                self.primary_name,
                trigger,
                type(error).__name__,
                self.fallback_name,
            )
            try:
                result = self.fallback_call()
            except Exception:
                self.provenance = {
                    "configured_provider": self.primary_name,
                    "effective_provider": self.fallback_name,
                    "fallback_used": True,
                    "fallback_trigger": trigger,
                    "primary_error_type": type(error).__name__,
                    "primary_retry_count": len(getattr(self.primary, "validation", [])),
                    "fallback_success": False,
                }
                raise
            self.provenance = {
                "configured_provider": self.primary_name,
                "effective_provider": self.fallback_name,
                "fallback_used": True,
                "fallback_trigger": trigger,
                "primary_error_type": type(error).__name__,
                "primary_retry_count": len(getattr(self.primary, "validation", [])),
                "fallback_success": True,
            }
            return result


class FallbackOCRProvider(_FallbackProvider, OCRProvider):
    def detect_text(self, frames: List[Frame]) -> List[Segment]:
        self.fallback_call = lambda: self.fallback.detect_text(frames)
        return self._run(lambda: self.primary.detect_text(frames))

    def read_confirmed_tracks(
        self,
        regions: List[SubtitleRegion],
        frames: List[Frame],
        video_width: int,
        video_height: int,
    ) -> List[Segment]:
        read_track = getattr(self.primary, "read_confirmed_tracks", None)
        if not callable(read_track):
            raise NotImplementedError("Primary OCR provider does not support read_confirmed_tracks")
        self.fallback_call = lambda: (
            self.fallback.read_confirmed_tracks(regions, frames, video_width, video_height)
            if hasattr(self.fallback, "read_confirmed_tracks")
            else []
        )
        return self._run(lambda: read_track(regions, frames, video_width, video_height))


class FallbackSpeechProvider(_FallbackProvider, SpeechProvider):
    def transcribe(self, audio_path: str, source_language: str) -> List[Segment]:
        self.fallback_call = lambda: self.fallback.transcribe(audio_path, source_language)
        return self._run(lambda: self.primary.transcribe(audio_path, source_language))
