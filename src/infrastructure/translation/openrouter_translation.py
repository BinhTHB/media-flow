"""OpenRouter translation adapter with multi-account key rotation and model fallback.

The adapter intentionally handles only text translation. Visual subtitle and audio
pipelines create separate instances with different ``purpose`` values, so they can
translate the same source independently without sharing state or wording policy.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from typing import List, Optional, Sequence

from src.core.contracts import TranslationPort

logger = logging.getLogger(__name__)

DEFAULT_FREE_MODELS = (
    # Keep the primary tier lightweight for simple contextual translation.
    # Free endpoint availability changes over time, so retain several providers.
    "inclusionai/ling-3.0-flash:free",
    "google/gemma-4-26b-a4b-it:free",
    "minimax/minimax-m2.7:free",
)


def _split_values(raw: str | None) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in re.split(r"[,;\n]+", raw) if item.strip()]


def configured_openrouter_keys() -> List[str]:
    """Return configured OpenRouter keys without exposing/logging their values."""
    values: List[str] = []
    values.extend(_split_values(os.getenv("OPENROUTER_API_KEYS")))
    single = os.getenv("OPENROUTER_API_KEY", "").strip()
    if single:
        values.append(single)
    for name in sorted(os.environ):
        if name.startswith("OPENROUTER_API_KEY_"):
            value = os.getenv(name, "").strip()
            if value:
                values.append(value)
    # Preserve order while avoiding repeated attempts with the same key.
    return list(dict.fromkeys(values))


class _OpenRouterHTTPError(RuntimeError):
    def __init__(self, status: int, detail: str, retry_after: float | None = None) -> None:
        super().__init__(f"HTTP {status}: {detail[:300]}")
        self.status = status
        self.retry_after = retry_after


class OpenRouterTranslationAdapter(TranslationPort):
    endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_keys: Optional[Sequence[str]] = None,
        models: Optional[Sequence[str]] = None,
        *,
        purpose: str = "subtitle",
        timeout_seconds: float = 90.0,
        chunk_size: Optional[int] = None,
    ) -> None:
        self.api_keys = [key.strip() for key in (api_keys or configured_openrouter_keys()) if key and key.strip()]
        env_models = _split_values(os.getenv("OPENROUTER_TRANSLATION_MODELS"))
        self.models = list(models or env_models or DEFAULT_FREE_MODELS)
        self.purpose = purpose if purpose in {"subtitle", "spoken"} else "subtitle"
        self.timeout_seconds = timeout_seconds
        self.chunk_size = chunk_size or (80 if self.purpose == "subtitle" else 40)
        self._cursor = 0
        self._disabled_key_indices: set[int] = set()
        self._cooldown_until: dict[int, float] = {}
        self._default_429_cooldown = max(10.0, float(os.getenv("OPENROUTER_429_COOLDOWN_SECONDS", "900")))
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return bool(self.api_keys and self.models)

    def translate_texts(
        self,
        texts: List[str],
        target_language: str = "vi",
        source_language: str = "auto",
    ) -> List[str]:
        if not texts:
            return []
        if not self.api_keys:
            raise RuntimeError("OpenRouter translation requires OPENROUTER_API_KEY(S).")
        if not self.models:
            raise RuntimeError("OpenRouter translation has no configured models.")

        translated: List[str] = []
        for start in range(0, len(texts), self.chunk_size):
            chunk = texts[start : start + self.chunk_size]
            translated.extend(
                self._translate_chunk(
                    chunk,
                    target_language=target_language,
                    source_language=source_language,
                )
            )
        return translated

    def _translate_chunk(
        self,
        texts: List[str],
        *,
        target_language: str,
        source_language: str,
    ) -> List[str]:
        payload_base = self._request_payload(
            texts,
            target_language=target_language,
            source_language=source_language,
        )
        errors: List[str] = []

        with self._lock:
            start_cursor = self._cursor % len(self.api_keys)

        # Model fallback is intentional: free endpoints can be temporarily saturated.
        for model in self.models:
            model_fatal = False
            for offset in range(len(self.api_keys)):
                key_index = (start_cursor + offset) % len(self.api_keys)
                with self._lock:
                    if key_index in self._disabled_key_indices:
                        continue
                    cooldown_until = self._cooldown_until.get(key_index, 0.0)
                if cooldown_until > time.monotonic():
                    continue
                key = self.api_keys[key_index]
                payload = dict(payload_base)
                payload["model"] = model
                try:
                    content = self._post(payload, key)
                    result = self._parse_response(content, texts)
                    self._validate_result(result, texts)
                    with self._lock:
                        self._cursor = (key_index + 1) % len(self.api_keys)
                    return result
                except Exception as exc:  # rotate model/key on quota/capacity/format failure
                    safe_error = self._safe_error(exc)
                    errors.append(f"{model} key#{key_index + 1}: {safe_error}")
                    if isinstance(exc, _OpenRouterHTTPError) and exc.status in {401, 403}:
                        # Authentication belongs to the key/account, not the model.
                        # Keep it disabled for this adapter instance so later batches
                        # do not waste RPD/latency retrying a known-bad credential.
                        with self._lock:
                            self._disabled_key_indices.add(key_index)
                    elif isinstance(exc, _OpenRouterHTTPError) and exc.status == 429 and self._is_upstream_model_rate_limit(safe_error):
                        # OpenRouter can return 429 because the selected free model/provider
                        # is saturated. That is model capacity, not account quota: keep the
                        # key healthy and immediately try the next model.
                        model_fatal = True
                    elif isinstance(exc, _OpenRouterHTTPError) and exc.status in {402, 429}:
                        cooldown = exc.retry_after or (3600.0 if exc.status == 402 else self._default_429_cooldown)
                        with self._lock:
                            self._cooldown_until[key_index] = time.monotonic() + max(1.0, cooldown)
                    elif self._is_model_request_error(safe_error):
                        # 400/404/422 is normally model/payload compatibility; trying
                        # the same model with every account key adds no value.
                        model_fatal = True
                    if model_fatal:
                        break
            with self._lock:
                if len(self._disabled_key_indices) >= len(self.api_keys):
                    break

        summary = " | ".join(errors[-6:])
        raise RuntimeError(f"OpenRouter translation exhausted configured model/key pool. {summary}")

    def _request_payload(
        self,
        texts: List[str],
        *,
        target_language: str,
        source_language: str,
    ) -> dict:
        rows = [{"id": idx, "text": text} for idx, text in enumerate(texts)]
        if self.purpose == "spoken":
            policy = (
                "These are consecutive speech segments from one video scene. Translate them into natural "
                f"spoken {target_language}. Preserve every id and every meaning-bearing detail; never omit short "
                "utterances, greetings, names, numbers, negation, objects, or repetitions that affect meaning. "
                "Use neighboring segments only to disambiguate meaning; NEVER copy words or meaning from a neighboring "
                "segment into the current id unless the current source text itself expresses them. Keep wording concise "
                "enough for dubbing and natural when spoken aloud. Do not merge or split ids."
            )
        else:
            policy = (
                "These are consecutive on-screen subtitle events from one video scene. First infer the scene/topic/domain "
                "from the whole batch, then translate each id into natural, concise "
                f"{target_language} captions. Preserve every id, tone, name, number, and meaning-bearing detail. "
                "Resolve ambiguous words and jargon using the inferred domain instead of the everyday literal meaning; "
                "for example, in Chinese stock-market context 国家队 refers to state-backed market institutions/funds, "
                "not a sports national team. Keep terminology consistent across neighboring ids, transliterate or retain "
                "proper names appropriately, and prefer short readable phrasing suitable for at most two subtitle lines. "
                "Do not merge or split ids, and never copy content from a neighboring id unless it is needed only to "
                "disambiguate the current id."
            )
        language_hint = (
            "Detect the source language automatically."
            if source_language == "auto"
            else f"The source language is {source_language}."
        )
        prompt = (
            f"{policy}\n{language_hint}\n"
            "Return ONLY JSON in this exact shape: "
            '{"translations":[{"id":0,"text":"..."}]}. '
            "The translations array must contain exactly one entry for every input id, in the same order.\n\n"
            f"INPUT={json.dumps(rows, ensure_ascii=False)}"
        )
        return {
            "messages": [
                {"role": "system", "content": "You are a careful professional audiovisual translator."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.15,
            "response_format": {"type": "json_object"},
        }

    def _post(self, payload: dict, api_key: str) -> str:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "MediaFlow"),
        }
        referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
        if referer:
            headers["HTTP-Referer"] = referer
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            retry_after = None
            try:
                raw_retry = exc.headers.get("Retry-After") if exc.headers else None
                retry_after = float(raw_retry) if raw_retry else None
            except (TypeError, ValueError):
                retry_after = None
            raise _OpenRouterHTTPError(exc.code, detail, retry_after=retry_after) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"network error: {exc.reason}") from exc

        choices = body.get("choices") if isinstance(body, dict) else None
        if not choices or not isinstance(choices, list):
            raise RuntimeError("response contains no choices")
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("response contains no text content")
        return content.strip()

    @staticmethod
    def _parse_response(content: str, originals: List[str]) -> List[str]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise RuntimeError("model returned invalid JSON") from exc
        rows = payload.get("translations") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise RuntimeError("JSON is missing translations array")
        by_id: dict[int, str] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                idx = int(row.get("id"))
            except (TypeError, ValueError):
                continue
            text = str(row.get("text") or "").strip()
            if text:
                by_id[idx] = text
        return [by_id.get(idx, "") for idx in range(len(originals))]

    @staticmethod
    def _validate_result(result: List[str], originals: List[str]) -> None:
        if len(result) != len(originals):
            raise RuntimeError(f"translation count mismatch {len(result)} != {len(originals)}")
        missing = [idx for idx, value in enumerate(result) if not value.strip()]
        if missing:
            raise RuntimeError(f"translation missing ids: {missing[:8]}")

    @staticmethod
    def _is_auth_error(text: str) -> bool:
        return "HTTP 401" in text or "HTTP 403" in text

    @staticmethod
    def _is_model_request_error(text: str) -> bool:
        return any(code in text for code in ("HTTP 400", "HTTP 404", "HTTP 422"))

    @staticmethod
    def _is_upstream_model_rate_limit(text: str) -> bool:
        lowered = text.lower()
        return "temporarily rate-limited upstream" in lowered or "upstream rate" in lowered

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        # Errors may echo request details; aggressively remove Bearer-like tokens.
        text = str(exc).replace("\n", " ")
        text = re.sub(r"sk-or-v1-[A-Za-z0-9_-]+", "<redacted>", text)
        return text[:320]
