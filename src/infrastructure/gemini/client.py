from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from src.infrastructure.gemini.contracts import (
    GeminiRequest,
    GeminiResponse,
    GeminiUsage,
    GeminiModelInfo,
    GeminiTransport,
    GEMINI_API_BASE,
    GEMINI_API_VERSION,
    GENERATE_CONTENT_METHOD,
    DEFAULT_MODEL,
    MODE_ROLE_PRIMARY,
    MODE_ROLE_ESCALATION,
)
from src.infrastructure.gemini.exceptions import (
    GeminiError,
    GeminiNetworkError,
    GeminiAPIError,
    GeminiRateLimitError,
    GeminiAuthError,
    GeminiValidationError,
)
from src.infrastructure.gemini.request_scheduler import RequestScheduler
from src.infrastructure.gemini.response_cache import ResponseCache
from src.infrastructure.gemini.usage_recorder import UsageRecorder, UsageRecord
from src.infrastructure.gemini.file_manager import FileManager


class GeminiClient:
    """Lazy-init Gemini REST client (stdlib urllib). Integrates scheduler, cache, usage."""

    def __init__(self, *, model: str = DEFAULT_MODEL, api_key: Optional[str] = None,
                 base_url: str = f"{GEMINI_API_BASE}/{GEMINI_API_VERSION}",
                 transport: Optional[GeminiTransport] = None,
                 scheduler: Optional[RequestScheduler] = None,
                 cache: Optional[ResponseCache] = None,
                 usage_recorder: Optional[UsageRecorder] = None,
                 file_manager: Optional[FileManager] = None,
                 role: str = MODE_ROLE_PRIMARY, timeout: float = 60.0) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._transport: GeminiTransport = transport or _UrllibTransport()
        self._scheduler = scheduler
        self._cache = cache
        self._usage = usage_recorder
        self._files = file_manager
        self._role = role
        self._timeout = timeout

    # --- lazy key resolution ---
    def _resolve_api_key(self) -> str:
        key = self._api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise GeminiAuthError("GEMINI_API_KEY is not set; provide api_key or set the environment variable.")
        return key

    def _model_url(self, action: str) -> str:
        key = self._resolve_api_key()
        return f"{self._base_url}/models/{self._model}:{action}?key={key}"

    def _models_url(self, model: Optional[str] = None) -> str:
        key = self._resolve_api_key()
        if model:
            return f"{self._base_url}/models/{model}?key={key}"
        return f"{self._base_url}/models?key={key}"

    # --- generate content ---
    def generate_content(self, request: GeminiRequest, *, use_cache: bool = True) -> GeminiResponse:
        request.model = request.model or self._model
        cache_key = None
        if use_cache and self._cache is not None:
            cache_key = self._cache.make_key(
                model=request.model,
                role=self._role,
                contents=request.contents,
                generation_config=request.generation_config,
                system_instruction=request.system_instruction,
                prompt_version=request.prompt_version,
                schema_version=request.schema_version,
                api_surface=request.api_surface,
                source_kind=request.source_kind,
                media_sha256=request.media_sha256,
            )
            cached = self._cache.get(cache_key)
            if cached is not None:
                resp = GeminiResponse.from_dict(cached)
                resp.cache_hit = True
                self._record_usage(request, resp, cache_hit=True, retry_count=0)
                return resp

        body = self._build_body(request)
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        attempt = 0
        if self._scheduler is not None:
            self._scheduler.acquire(tokens=self._estimate_tokens(data))
        try:
            while True:
                try:
                    raw = self._transport.send(
                        "POST", self._model_url(GENERATE_CONTENT_METHOD),
                        headers=headers, data=data, timeout=self._timeout,
                    )
                    resp = self._parse_response(raw, request)
                    if cache_key is not None and self._cache is not None:
                        self._cache.set(cache_key, resp.to_dict())
                    self._record_usage(request, resp, cache_hit=False, retry_count=attempt)
                    return resp
                except GeminiRateLimitError as exc:
                    if self._scheduler is not None and self._scheduler.should_retry(attempt):
                        self._scheduler.on_rate_limited(exc.retry_after)
                        if self._usage is not None:
                            self._usage.record_rate_limit(request.model or self._model)
                        attempt += 1
                        continue
                    raise
                except GeminiError:
                    raise
        finally:
            if self._scheduler is not None:
                self._scheduler.release()

    def _build_body(self, request: GeminiRequest) -> Dict[str, Any]:
        body: Dict[str, Any] = {"contents": request.contents}
        if request.generation_config is not None:
            body["generationConfig"] = request.generation_config
        if request.system_instruction is not None:
            body["systemInstruction"] = {"parts": [{"text": request.system_instruction}]}
        if request.safety_settings is not None:
            body["safetySettings"] = request.safety_settings
        return body

    def _parse_response(self, raw: Dict[str, Any], request: GeminiRequest) -> GeminiResponse:
        if not isinstance(raw, dict):
            raise GeminiValidationError("Malformed Gemini response: expected a JSON object.")
        if raw.get("error"):
            err = raw["error"]
            msg = err.get("message", "Unknown Gemini API error") if isinstance(err, dict) else str(err)
            raise GeminiAPIError(f"Gemini API error: {msg}")
        candidates = raw.get("candidates")
        if not candidates:
            raise GeminiValidationError("Gemini response contained no candidates.")
        first = candidates[0]
        parts = first.get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        finish_reason = first.get("finishReason")
        safety = first.get("safetyRatings", [])
        usage = GeminiUsage.from_api(raw.get("usageMetadata"))
        return GeminiResponse(
            text=text,
            model=self._model,
            finish_reason=finish_reason,
            safety_ratings=safety,
            usage=usage,
            request_id=request.request_id,
            raw=raw,
        )

    # --- model client protocol (for validator) ---
    def get_model(self, model: str) -> GeminiModelInfo:
        raw = self._transport.send("GET", self._models_url(model), headers={}, timeout=self._timeout)
        if not isinstance(raw, dict):
            raise GeminiValidationError("Malformed model info response.")
        return GeminiModelInfo.from_api_dict(raw)

    def list_models(self) -> List[GeminiModelInfo]:
        raw = self._transport.send("GET", self._models_url(), headers={}, timeout=self._timeout)
        models = raw.get("models", []) if isinstance(raw, dict) else []
        return [GeminiModelInfo.from_api_dict(m) for m in models]

    # --- helpers ---
    def _estimate_tokens(self, body: bytes) -> int:
        return max(1, len(body) // 4)

    def _record_usage(self, request: GeminiRequest, resp: GeminiResponse, *,
                      cache_hit: bool, retry_count: int) -> None:
        if self._usage is None:
            return
        self._usage.record(UsageRecord(
            model=request.model or self._model,
            role=self._role,
            prompt_tokens=resp.usage.prompt_token_count,
            candidates_tokens=resp.usage.candidates_token_count,
            total_tokens=resp.usage.total_token_count,
            request_id=request.request_id,
            cache_hit=cache_hit,
            retry_count=retry_count,
            is_escalation=(self._role == MODE_ROLE_ESCALATION),
            source_kind=request.source_kind,
            prompt_version=request.prompt_version,
            schema_version=request.schema_version,
            api_surface=request.api_surface,
            media_sha256=request.media_sha256,
        ))


class _UrllibTransport:
    def send(self, method: str, url: str, *, headers: Dict[str, str],
             data: Optional[bytes] = None, timeout: float = 60.0) -> Dict[str, Any]:
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_bytes = resp.read()
        except urllib.error.HTTPError as exc:
            _raise_for_http_error(exc)
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise GeminiNetworkError(f"Gemini request failed: {reason}") from exc
        try:
            return json.loads(raw_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise GeminiValidationError(f"Invalid JSON response from Gemini: {exc}") from exc


def _raise_for_http_error(exc: urllib.error.HTTPError) -> None:
    status = exc.code
    try:
        payload = json.loads(exc.read().decode("utf-8"))
        msg = payload.get("error", {}).get("message", exc.reason) if isinstance(payload, dict) else str(exc.reason)
    except Exception:
        msg = str(exc.reason)
    if status == 429:
        raise GeminiRateLimitError(f"Gemini rate limit exceeded (429): {msg}", status_code=429,
                                   retry_after=_parse_retry_after(exc))
    if status in (401, 403):
        raise GeminiAuthError(f"Gemini authentication failed ({status}): {msg}", status_code=status)
    raise GeminiAPIError(f"Gemini API error ({status}): {msg}", status_code=status)


def _parse_retry_after(exc: urllib.error.HTTPError) -> Optional[float]:
    try:
        header = exc.headers.get("Retry-After")
        if header:
            return float(header)
    except Exception:
        pass
    return None
