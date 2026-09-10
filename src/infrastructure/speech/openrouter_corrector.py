"""Evidence-based contextual ASR correction through OpenRouter.

The corrector is intentionally selective to protect rate limits:
- Primary ASR text is trusted by default.
- A secondary ASR is called only when suspicious transcript features are found.
- Only suspicious segments are eligible for text replacement.
- Segment ids/timestamps are never changed.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import List, Sequence

from src.core.contracts import ASRCorrectionPort, ASRPort
from src.core.models.speech_segment import SpeechSegment
from src.infrastructure.translation.openrouter_translation import (
    DEFAULT_FREE_MODELS,
    _split_values,
    configured_openrouter_keys,
)

logger = logging.getLogger(__name__)

_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")


class OpenRouterASRCorrector(ASRCorrectionPort):
    endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_keys: Sequence[str] | None = None,
        models: Sequence[str] | None = None,
        secondary_provider: ASRPort | None = None,
        timeout_seconds: float = 90.0,
    ) -> None:
        self.api_keys = [k.strip() for k in (api_keys or configured_openrouter_keys()) if k and k.strip()]
        env_models = _split_values(os.getenv("OPENROUTER_ASR_CORRECTION_MODELS"))
        self.models = list(models or env_models or DEFAULT_FREE_MODELS)
        self.secondary_provider = secondary_provider
        self.timeout_seconds = timeout_seconds
        self._cursor = 0

    def correct_segments(
        self,
        segments: List[SpeechSegment],
        language: str = "auto",
        audio_path: str | None = None,
    ) -> List[SpeechSegment]:
        if not segments or not self.api_keys or not self.models:
            return segments

        suspicious = self._suspicious_indices(segments, language)
        if not suspicious:
            return segments

        alternatives: dict[int, str] = {}
        if self.secondary_provider is not None and audio_path:
            try:
                secondary = self.secondary_provider.transcribe(audio_path, language=language)
                alternatives = self._align_alternatives(segments, secondary, suspicious)
            except Exception as exc:
                logger.warning("Secondary ASR failed; retaining primary transcript: %s", exc)
                return segments

        # Do not ask an LLM to invent a correction without independent audio evidence.
        if not any(alternatives.get(index, "").strip() for index in suspicious):
            return segments

        payload_base = self._build_payload(segments, suspicious, alternatives, language)
        errors: list[str] = []
        for model in self.models:
            for offset in range(len(self.api_keys)):
                key_index = (self._cursor + offset) % len(self.api_keys)
                try:
                    payload = dict(payload_base)
                    payload["model"] = model
                    text = self._post(payload, self.api_keys[key_index])
                    replacements = self._parse_response(text, suspicious)
                    # A response is not a successful correction if it preserves the
                    # exact suspicious failure mode while independent secondary ASR
                    # provides a native-script alternative. Reject it and continue
                    # to the next model/key instead of making correction stochastic.
                    for idx, corrected_text in replacements.items():
                        primary_text = segments[idx].source_text or ""
                        secondary_text = alternatives.get(idx, "") or ""
                        primary_mixed = bool(_CJK_RE.search(primary_text) and _LATIN_TOKEN_RE.search(primary_text))
                        secondary_native = bool(_CJK_RE.search(secondary_text) and not _LATIN_TOKEN_RE.search(secondary_text))
                        corrected_mixed = bool(_CJK_RE.search(corrected_text) and _LATIN_TOKEN_RE.search(corrected_text))
                        if primary_mixed and secondary_native and corrected_mixed:
                            raise RuntimeError(
                                "consensus preserved suspicious mixed CJK+Latin despite native-script secondary evidence"
                            )
                    self._cursor = (key_index + 1) % len(self.api_keys)
                    for idx, corrected_text in replacements.items():
                        if corrected_text:
                            segments[idx].source_text = corrected_text
                    return segments
                except Exception as exc:
                    errors.append(f"{model} key#{key_index + 1}: {str(exc)[:180]}")

        logger.warning("ASR consensus failed open; primary transcript retained. %s", " | ".join(errors[-4:]))
        return segments

    @staticmethod
    def _suspicious_indices(segments: List[SpeechSegment], language: str) -> List[int]:
        result: list[int] = []
        for idx, segment in enumerate(segments):
            text = segment.source_text or ""
            # Mixed CJK + Latin tokens are worth validating, but not automatically replacing.
            # This catches errors such as Chinese 快递/外卖 being decoded as "Wi-Fi".
            if _CJK_RE.search(text) and _LATIN_TOKEN_RE.search(text):
                result.append(idx)
                continue
            if "�" in text or "<?>" in text:
                result.append(idx)
        return result

    @staticmethod
    def _align_alternatives(
        primary: List[SpeechSegment],
        secondary: List[SpeechSegment],
        suspicious: Sequence[int],
    ) -> dict[int, str]:
        alternatives: dict[int, str] = {}
        for idx in suspicious:
            item = primary[idx]
            overlaps: list[tuple[float, SpeechSegment]] = []
            for candidate in secondary:
                overlap = min(item.end_time, candidate.end_time) - max(item.start_time, candidate.start_time)
                if overlap > 0.05:
                    overlaps.append((overlap, candidate))
            overlaps.sort(key=lambda pair: (pair[1].start_time, -pair[0]))
            text = " ".join(seg.source_text.strip() for _, seg in overlaps if seg.source_text.strip()).strip()
            if text and text != item.source_text.strip():
                alternatives[idx] = text
        return alternatives

    @staticmethod
    def _build_payload(
        segments: List[SpeechSegment],
        suspicious: Sequence[int],
        alternatives: dict[int, str],
        language: str,
    ) -> dict:
        scene = [
            {"id": i, "start": round(seg.start_time, 3), "end": round(seg.end_time, 3), "primary": seg.source_text}
            for i, seg in enumerate(segments)
        ]
        candidates = [
            {
                "id": idx,
                "primary": segments[idx].source_text,
                "secondary": alternatives.get(idx, ""),
            }
            for idx in suspicious
            if alternatives.get(idx, "")
        ]
        prompt = (
            "Resolve ASR disagreements for one continuous video scene.\n"
            f"Language hint: {language}.\n"
            "Primary timestamps/segmentation are authoritative. Only choose the most likely spoken TEXT.\n"
            "Use phonetic plausibility, grammar, and surrounding scene context.\n"
            "Rules:\n"
            "- Do NOT translate. Keep the source language/script.\n"
            "- Only ids listed in candidates may change.\n"
            "- Prefer an existing primary or secondary candidate; make a tiny correction only if both clearly contain the same obvious recognition error.\n"
            "- Mixed CJK + unexpected Latin tokens are a known ASR failure mode. If primary contains such a token but secondary provides a natural native-script phrase that fits the surrounding verb/context, prefer secondary unless the Latin token is clearly a supported proper name or technical term.\n"
            "- Preserve names, numbers, negation, repetitions, slang and short utterances.\n"
            "- If evidence is genuinely ambiguous after contextual/grammatical comparison, keep primary exactly.\n"
            "Return strict JSON only: {\"corrections\":[{\"id\":5,\"text\":\"...\"}]}.\n\n"
            f"Full scene: {json.dumps(scene, ensure_ascii=False)}\n"
            f"Disagreements: {json.dumps(candidates, ensure_ascii=False)}"
        )
        return {
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "Evidence-based ASR consensus. JSON only."},
                {"role": "user", "content": prompt},
            ],
        }

    def _post(self, payload: dict, api_key: str) -> str:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
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
        request = urllib.request.Request(self.endpoint, data=body, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload_out = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail[:300]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"network error: {exc.reason}") from exc

        choices = payload_out.get("choices") if isinstance(payload_out, dict) else None
        if not choices:
            raise RuntimeError("OpenRouter ASR consensus returned no choices")
        content = choices[0].get("message", {}).get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("OpenRouter ASR consensus returned empty content")
        return content.strip()

    @staticmethod
    def _parse_response(content: str, suspicious: Sequence[int]) -> dict[int, str]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lstrip().startswith("json"):
                cleaned = cleaned.lstrip()[4:].lstrip()
        data = json.loads(cleaned)
        items = data.get("corrections") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise RuntimeError("ASR consensus response missing corrections[]")

        allowed = set(suspicious)
        result: dict[int, str] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            idx = item.get("id")
            text = item.get("text")
            if isinstance(idx, int) and idx in allowed and isinstance(text, str) and text.strip():
                result[idx] = text.strip()
        if not result:
            raise RuntimeError("ASR consensus returned no usable correction")
        return result
