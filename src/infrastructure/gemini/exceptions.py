from __future__ import annotations

import re
from typing import Any, Optional

from src.shared.exceptions import ProviderError

_API_KEY_PATTERN = re.compile(r"AIzaSy[A-Za-z0-9_-]{33}")
_QUERY_KEY_PATTERN = re.compile(r"([\?&]key=)[^&\s\"\']+")


def _redact(value: str) -> str:
    if not isinstance(value, str):
        return value
    value = _API_KEY_PATTERN.sub("[REDACTED_API_KEY]", value)
    value = _QUERY_KEY_PATTERN.sub("\\1[REDACTED]", value)
    return value


class GeminiError(ProviderError):
    """Base exception for Gemini infrastructure; all Gemini errors subclass ProviderError."""

    def __init__(self, message: str, *args: Any) -> None:
        super().__init__(_redact(message), *args)


class GeminiNetworkError(GeminiError):
    """Network-level failure contacting the Gemini API."""


class GeminiAPIError(GeminiError):
    """Gemini API returned an error status. Never embeds raw response bodies."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class GeminiRateLimitError(GeminiAPIError):
    """Gemini API returned HTTP 429. Carries optional Retry-After hint (seconds)."""

    def __init__(self, message: str, status_code: int = 429, retry_after: Optional[float] = None) -> None:
        super().__init__(message, status_code=status_code)
        self.retry_after = retry_after


class GeminiAuthError(GeminiAPIError):
    """Authentication/authorization failure (HTTP 401/403)."""


class GeminiValidationError(GeminiError):
    """Request payload or structured output validation failure."""


class GeminiModelValidationError(GeminiError):
    """Model is unavailable, retired, or does not support generateContent."""


class GeminiCacheError(GeminiError):
    """Response cache read/write failure."""


class GeminiSchedulerError(GeminiError):
    """Request scheduler failure."""


class GeminiUsageError(GeminiError):
    """Usage recording/serialization failure."""


class GeminiFileError(GeminiError):
    """File upload/management failure."""

class ProviderConfigurationError(GeminiError):
    """Configured Gemini infrastructure cannot be used safely."""

class ProviderRateLimitError(GeminiRateLimitError):
    """A locally enforced provider quota has been exhausted."""


class GeminiConfigurationError(ProviderConfigurationError):
    """Model configuration is invalid for use (legacy/deprecated/retired/unknown)."""
    def __init__(self, message: str, *args: Any) -> None:
        super().__init__(_redact(message), *args)
