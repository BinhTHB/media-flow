from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

DEFAULT_MODEL = "gemini-3.5-flash-lite"
ESCALATION_MODEL = "gemini-3.5-flash"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com"
GEMINI_API_VERSION = "v1beta"
GENERATE_CONTENT_METHOD = "generateContent"
COUNT_TOKENS_METHOD = "countTokens"

MODEL_STAGE_STABLE = "STABLE"
MODEL_STAGE_PREVIEW = "PREVIEW"
MODEL_STAGE_LEGACY = "LEGACY"
MODEL_STAGE_DEPRECATED = "DEPRECATED"
MODEL_STAGE_RETIRED = "RETIRED"
MODEL_STAGE_UNSPECIFIED = "MODEL_STAGE_UNSPECIFIED"
MODEL_STAGE_UNSTABLE_EXPERIMENTAL = "UNSTABLE_EXPERIMENTAL"

MODEL_STAGE_OK = {MODEL_STAGE_STABLE, MODEL_STAGE_PREVIEW}
MODEL_STAGE_FAIL = {
    MODEL_STAGE_LEGACY,
    MODEL_STAGE_DEPRECATED,
    MODEL_STAGE_RETIRED,
    MODEL_STAGE_UNSPECIFIED,
    MODEL_STAGE_UNSTABLE_EXPERIMENTAL,
}

MODE_ROLE_PRIMARY = "primary"
MODE_ROLE_ESCALATION = "escalation"

@dataclass
class GeminiUsage:

    prompt_token_count: int = 0
    candidates_token_count: int = 0
    total_token_count: int = 0
    model_stage: str = "unknown"

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_token_count": self.prompt_token_count,
            "candidates_token_count": self.candidates_token_count,
            "total_token_count": self.total_token_count,
            "model_stage": self.model_stage,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeminiUsage":
        return cls(
            prompt_token_count=int(data.get("prompt_token_count", 0) or 0),
            candidates_token_count=int(data.get("candidates_token_count", 0) or 0),
            total_token_count=int(data.get("total_token_count", 0) or 0),
            model_stage=str(data.get("model_stage", "unknown") or "unknown"),
        )

    @classmethod
    def from_api(cls, data: Optional[Dict[str, Any]]) -> "GeminiUsage":
        if not data:
            return cls()
        return cls.from_dict(data)


@dataclass


@dataclass
class GeminiProvenance:
    model: str
    model_stage: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "model_stage": self.model_stage,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeminiProvenance":
        return cls(
            model=data.get("model", ""),
            model_stage=data.get("model_stage", "unknown"),
        )

@dataclass
class GeminiResponse:
    text: str
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    safety_ratings: List[Dict[str, Any]] = field(default_factory=list)
    usage: GeminiUsage = field(default_factory=GeminiUsage)
    request_id: Optional[str] = None
    cache_hit: bool = False
    provenance: Optional[GeminiProvenance] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    model_stage: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "safety_ratings": self.safety_ratings,
            "usage": self.usage.to_dict(),
            "request_id": self.request_id,
            "cache_hit": self.cache_hit,
            "model_stage": self.model_stage,
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeminiResponse":
        return cls(
            text=data.get("text", ""),
            model=data.get("model"),
            finish_reason=data.get("finish_reason"),
            safety_ratings=list(data.get("safety_ratings", []) or []),
            usage=GeminiUsage.from_dict(data.get("usage", {}) or {}),
            request_id=data.get("request_id"),
            cache_hit=bool(data.get("cache_hit", False)),
            model_stage=str(data.get("model_stage", "unknown") or "unknown"),
            provenance=(
                GeminiProvenance.from_dict(data["provenance"])
                if data.get("provenance") else None
            ),
        )


@dataclass
class GeminiRequest:
    contents: Any
    model: Optional[str] = None
    generation_config: Optional[Dict[str, Any]] = None
    system_instruction: Optional[str] = None
    safety_settings: Optional[List[Dict[str, Any]]] = None
    request_id: Optional[str] = None
    prompt_version: Optional[str] = None
    schema_version: Optional[str] = None
    source_kind: Optional[str] = None
    media_sha256: Optional[str] = None
    api_surface: Optional[str] = None
    predicted_tokens: Optional[int] = None
    role: str = MODE_ROLE_PRIMARY

    def normalized_contents(self) -> Any:
        return self.contents


@dataclass
class GeminiModelInfo:
    name: str
    base_model_id: Optional[str] = None
    version: Optional[str] = None
    display_name: Optional[str] = None
    description: Optional[str] = None
    supported_generation_methods: List[str] = field(default_factory=list)
    input_token_limit: Optional[int] = None
    output_token_limit: Optional[int] = None
    model_stage: Optional[str] = None
    retirement_time: Optional[str] = None

    @classmethod
    def from_api_dict(cls, data: Dict[str, Any]) -> "GeminiModelInfo":
        return cls(
            name=str(data.get("name", "")),
            base_model_id=data.get("baseModelId"),
            version=data.get("version"),
            display_name=data.get("displayName"),
            description=data.get("description"),
            supported_generation_methods=list(data.get("supportedGenerationMethods", []) or []),
            input_token_limit=data.get("inputTokenLimit"),
            output_token_limit=data.get("outputTokenLimit"),
            model_stage=data.get("modelStage"),
            retirement_time=data.get("retirementTime"),
        )

    def supports_generate_content(self) -> bool:
        return GENERATE_CONTENT_METHOD in (self.supported_generation_methods or [])


@runtime_checkable
class GeminiTransport(Protocol):
    """Low-level REST transport used by the client. Injectable for tests."""

    def send(self, method: str, url: str, *, headers: Dict[str, str],
             data: Optional[bytes] = None, timeout: float = 60.0) -> Dict[str, Any]:
        ...


@runtime_checkable
class GeminiModelClient(Protocol):
    """Client capability needed by the model validator. Implemented by GeminiClient."""

    def get_model(self, model: str) -> GeminiModelInfo:
        ...

    def list_models(self) -> List[GeminiModelInfo]:
        ...


@runtime_checkable
class GeminiFileClient(Protocol):
    """Client capability for file upload/delete used by the file manager scaffold."""

    def upload_file(self, path: str, *, mime_type: Optional[str] = None) -> Dict[str, Any]:
        ...

    def delete_file(self, file_name: str) -> None:
        ...

    def get_file(self, file_name: str) -> Dict[str, Any]:
        ...


def default_generation_config() -> Dict[str, Any]:
    return {}


def now_seconds() -> float:
    return time.time()
