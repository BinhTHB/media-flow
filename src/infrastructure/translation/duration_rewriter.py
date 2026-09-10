"""OpenRouter duration-aware spoken-copy rewriter.

This adapter does not translate. It shortens already-translated spoken copy only
when the natural TTS duration would require excessive time-stretch. The source
text is supplied as a semantic anchor so essential meaning is preserved.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Sequence

from src.core.contracts import SpokenRewritePort
from src.infrastructure.translation.openrouter_translation import (
    DEFAULT_FREE_MODELS,
    _split_values,
    configured_openrouter_keys,
)


class OpenRouterDurationRewriter(SpokenRewritePort):
    endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_keys: Sequence[str] | None = None,
        models: Sequence[str] | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.api_keys = [k.strip() for k in (api_keys or configured_openrouter_keys()) if k and k.strip()]
        env_models = _split_values(os.getenv("OPENROUTER_DURATION_REWRITE_MODELS"))
        self.models = list(models or env_models or DEFAULT_FREE_MODELS)
        self.timeout_seconds = timeout_seconds
        self._cursor = 0

    def rewrite_for_duration(
        self,
        text: str,
        source_text: str,
        target_duration: float,
        current_duration: float,
        target_language: str = "vi",
    ) -> str:
        text = (text or "").strip()
        source_text = (source_text or "").strip()
        if not text or not self.api_keys or not self.models:
            return text

        target_duration = max(0.20, float(target_duration))
        current_duration = max(0.01, float(current_duration))
        desired_ratio = min(0.96, target_duration / current_duration)
        # A character budget is only a hint. Vietnamese speaking duration does not
        # scale perfectly with characters, but this nudges simple free models away
        # from producing an equally long paraphrase.
        char_budget = max(3, int(round(len(text) * desired_ratio)))

        prompt = (
            "Rewrite ONE already-translated spoken line so it can be dubbed naturally inside a short time window.\n"
            f"Target language: {target_language}. Target speech window: {target_duration:.2f}s. "
            f"Current natural TTS duration: {current_duration:.2f}s. Approximate character target: <= {char_budget}.\n"
            f"Original source meaning anchor: {source_text}\n"
            f"Current translation: {text}\n\n"
            "Rules:\n"
            "- Preserve the essential meaning, negation, names, numbers and objects.\n"
            "- Remove politeness/filler/redundancy before removing semantic content.\n"
            "- Use natural conversational wording, not telegraphic fragments.\n"
            "- Do not add facts or meaning from neighboring lines; this is one isolated segment.\n"
            "- Do not translate back to the source language.\n"
            "- Prefer fewer syllables and common spoken words.\n"
            "Return strict JSON only: {\"text\":\"...\"}."
        )
        payload_base = {
            "messages": [
                {"role": "system", "content": "You are a concise audiovisual dubbing dialogue editor. JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }

        for model in self.models:
            for offset in range(len(self.api_keys)):
                key_index = (self._cursor + offset) % len(self.api_keys)
                payload = dict(payload_base)
                payload["model"] = model
                try:
                    content = self._post(payload, self.api_keys[key_index])
                    rewritten = self._parse(content)
                    if not rewritten:
                        continue
                    # Reject pathological expansion. A rewrite is useful only if it
                    # is materially shorter or the model chose a much terser wording.
                    if len(rewritten) > max(len(text) + 2, int(len(text) * 1.05)):
                        continue
                    self._cursor = (key_index + 1) % len(self.api_keys)
                    return rewritten
                except Exception:
                    continue
        return text

    def _post(self, payload: dict, api_key: str) -> str:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "MediaFlow/1.0",
        }
        app_title = os.getenv("OPENROUTER_APP_TITLE", "MediaFlow").strip()
        referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
        if app_title:
            headers["X-Title"] = app_title
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
            raise RuntimeError(f"HTTP {exc.code}: {detail[:240]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"network error: {exc.reason}") from exc

        choices = body.get("choices") if isinstance(body, dict) else None
        if not choices:
            raise RuntimeError("OpenRouter duration rewrite returned no choices")
        content = choices[0].get("message", {}).get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("OpenRouter duration rewrite returned empty content")
        return content.strip()

    @staticmethod
    def _parse(content: str) -> str:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lstrip().startswith("json"):
                cleaned = cleaned.lstrip()[4:].lstrip()
        data = json.loads(cleaned)
        value = data.get("text") if isinstance(data, dict) else None
        if not isinstance(value, str):
            return ""
        value = re.sub(r"\s+", " ", value).strip().strip('"')
        return value
