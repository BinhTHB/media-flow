class VideoTranslatorException(Exception):
    """Base exception for video translator project."""
    pass

class ProviderError(VideoTranslatorException):
    """Raised when an external provider fails."""
    pass

class ValidationError(VideoTranslatorException):
    """Raised when domain data is invalid."""
    pass

class UseCaseError(VideoTranslatorException):
    """Raised when a use case execution fails."""
    pass


class ProviderConfigurationError(ProviderError):
    pass

class ProviderRateLimitError(ProviderError):
    pass
