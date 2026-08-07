"""
MediaFlow Core Error Hierarchy

Pure Python custom exception classes for domain, feature, and infrastructure errors.
Zero external third-party dependencies.
"""

class MediaFlowError(Exception):
    """Base exception for all MediaFlow errors."""
    pass


class ValidationError(MediaFlowError):
    """Raised when input parameters, contract invariants, or data shapes are invalid."""
    pass


class ConfigurationError(MediaFlowError):
    """Raised when system or provider configuration is missing or invalid."""
    pass


class ProviderError(MediaFlowError):
    """Raised when an external infrastructure provider fails or throws an unhandled exception."""
    pass


class CredentialError(ProviderError):
    """Raised when required external credentials (e.g. GEMINI_API_KEY) are missing."""
    pass


class PipelineError(MediaFlowError):
    """Raised when an Audio or Visual pipeline stage fails to execute."""
    pass
