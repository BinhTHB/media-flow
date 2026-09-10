"""
Gemini Translation Infrastructure Adapter (Clean Architecture V2)

Implements TranslationPort using Gemini API directly via GeminiRuntime.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from src.core.contracts import TranslationPort
from src.infrastructure.gemini.contracts import GeminiRequest
from src.infrastructure.gemini.exceptions import GeminiAuthError, GeminiError
from src.infrastructure.gemini.runtime import GeminiRuntime

logger = logging.getLogger(__name__)


class GeminiTranslationAdapter(TranslationPort):
    def __init__(
        self,
        runtime: Optional[GeminiRuntime] = None,
        model: str = "gemini-3.5-flash-lite",
        purpose: str = "subtitle",
    ):
        self._runtime = runtime or GeminiRuntime.create()
        self.model = model
        self.purpose = purpose if purpose in {"subtitle", "spoken"} else "subtitle"

    def translate_texts(
        self, texts: List[str], target_language: str = "vi", source_language: str = "auto"
    ) -> List[str]:
        if not texts:
            return []

        if hasattr(self, "provider") and self.provider is not None:
            # Delegate to provider mock if injected in tests
            class DummySeg:
                def __init__(self, t):
                    self.text = t
                    self.translated_text = t
                    self.bbox = None

            segs = [DummySeg(t) for t in texts]
            res = self.provider.translate(segs, target_language=target_language)
            return [getattr(s, "translated_text", getattr(s, "text", "")) for s in res]

        chunk_size = 100
        translated_texts: List[str] = []
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i : i + chunk_size]
            chunk_translations = self._translate_chunk(chunk, target_language)
            translated_texts.extend(chunk_translations)

        return translated_texts

    def _translate_chunk(self, texts: List[str], target_language: str) -> List[str]:
        numbered_lines = "\n".join(f"{index}: {text}" for index, text in enumerate(texts))
        if self.purpose == "spoken":
            guidelines = (
                "1. These are consecutive speech segments from one video scene; use surrounding lines as context.\n"
                "2. Preserve every meaning-bearing detail, including short utterances, greetings, names, numbers, negation, objects, and meaningful repetition.\n"
                "3. Make the result natural when spoken aloud and concise enough for dubbing, but never omit content merely to shorten it.\n"
                "4. Do not merge or split numbered segments.\n"
            )
        else:
            guidelines = (
                "1. These are consecutive on-screen subtitle lines from one scene. Infer the topic/domain from the whole batch before translating each line.\n"
                "2. CRITICAL SEGMENT RULE: each numbered output may contain ONLY meaning present in that same numbered source line. Neighboring lines may disambiguate terminology but must NEVER donate clauses, numbers, objects, or details. If a source line is an incomplete fragment, keep it an incomplete but natural Vietnamese fragment; do not complete it using the next/previous line.\n"
                "3. Produce fluent, concise captions suitable for at most two subtitle lines.\n"
                "4. Resolve ambiguous words and jargon by domain, not by everyday literal meaning. In Chinese finance, 国家队 means state-backed market institutions/funds rather than a sports team; 游资 means speculative/hot-money capital or a speculator depending grammar; 燥热 describes physical sweltering/oi bức/nóng nực, never impatience or emotional excitement.\n"
                "5. Keep proper names, finance terminology, numbers, and recurring terms consistent. Prefer idiomatic Vietnamese over unnecessary English loanwords; for target Vietnamese, 巫师 should normally be 'Phù thủy' or 'Pháp sư', not 'Wizard'.\n"
                "6. Avoid word-for-word awkward or archaic translations (e.g. translate '拜拜' as 'Tạm biệt' or 'Chào nhé', NOT 'Biệt ly').\n"
                "7. Preserve original tone, emotion, meaning, and line count.\n"
            )
        prompt = (
            f"Translate each numbered video line into natural, conversational {target_language}.\n"
            "Translation guidelines:\n"
            f"{guidelines}"
            "7. Preserve every line number using the exact format 'i: translated_text'.\n"
            "8. Return only translated numbered lines with no explanations or extra quotes.\n\n"
            f"{numbered_lines}"
        )

        try:
            client = self._runtime.get_client(model=self.model)
            response = client.generate_content(
                GeminiRequest(
                    contents=[{"parts": [{"text": prompt}]}],
                    generation_config={"temperature": 0.3},
                    source_kind="translation",
                )
            )
            return self._parse_numbered_lines(response.text.strip(), texts)
        except GeminiAuthError as exc:
            logger.error("GEMINI_API_KEY environment variable is not set.")
            raise RuntimeError("GEMINI_API_KEY environment variable is not set.") from exc
        except GeminiError as exc:
            logger.error(f"Gemini translation failed: {exc}")
            raise RuntimeError(f"Gemini translation failed: {exc}") from exc

    @staticmethod
    def _parse_numbered_lines(content: str, original_texts: List[str]) -> List[str]:
        translated = {}
        for line in content.splitlines():
            index_text, separator, text = line.partition(":")
            if separator and index_text.strip().isdigit():
                translated[int(index_text.strip())] = text.strip()
        return [translated.get(index, original_texts[index]) for index in range(len(original_texts))]
