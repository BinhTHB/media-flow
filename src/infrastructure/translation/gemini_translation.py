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
    ):
        self._runtime = runtime or GeminiRuntime.create()
        self.model = model

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
        prompt = (
            f"Translate each numbered video subtitle line into natural, conversational {target_language}.\n"
            "Translation guidelines:\n"
            "1. Produce fluent, spoken-style subtitles suitable for video captions.\n"
            "2. Avoid word-for-word awkward or archaic translations (e.g. translate '拜拜' as 'Tạm biệt' or 'Chào nhé', NOT 'Biệt ly').\n"
            "3. Keep translations concise and appropriate for on-screen display.\n"
            "4. Preserve original tone, emotion, and line count.\n"
            "5. Preserve every line number using the exact format 'i: translated_text'.\n"
            "6. Return only translated numbered lines with no explanations or extra quotes.\n\n"
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
