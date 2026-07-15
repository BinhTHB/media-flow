from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.infrastructure.gemini.contracts import (
    GeminiModelClient,
    GENERATE_CONTENT_METHOD,
    MODEL_STAGE_FAIL,
)
from src.infrastructure.gemini.exceptions import GeminiModelValidationError


@dataclass
class ModelValidationResult:
    model: str
    available: bool
    supports_generate_content: bool
    stage: Optional[str]
    retirement_time: Optional[str]
    error: Optional[str] = None

    def require_valid(self) -> "ModelValidationResult":
        if not self.available:
            raise GeminiModelValidationError(f"Model '{self.model}' is not available or not found.")
        if not self.supports_generate_content:
            raise GeminiModelValidationError(
                f"Model '{self.model}' does not support generateContent."
            )
        return self


class ModelValidator:
    """Validates a model via the GeminiModelClient protocol (implemented by GeminiClient)."""

    def __init__(self, client: GeminiModelClient) -> None:
        self._client = client

    def validate(self, model: str) -> ModelValidationResult:
        try:
            info = self._client.get_model(model)
        except Exception as exc:
            return ModelValidationResult(
                model=model,
                available=False,
                supports_generate_content=False,
                stage=None,
                retirement_time=None,
                error=f"model lookup failed: {exc}",
            )
        stage = info.model_stage
        if stage in MODEL_STAGE_FAIL:
            return ModelValidationResult(
                model=model,
                available=False,
                supports_generate_content=info.supports_generate_content(),
                stage=stage,
                retirement_time=info.retirement_time,
                error=f"model stage '{stage}' is not usable",
            )
        return ModelValidationResult(
            model=model,
            available=True,
            supports_generate_content=info.supports_generate_content(),
            stage=stage,
            retirement_time=info.retirement_time,
        )

    def require_available(self, model: str) -> ModelValidationResult:
        return self.validate(model).require_valid()
