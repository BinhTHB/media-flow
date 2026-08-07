"""
Gemini Translation Infrastructure Adapter

Implements TranslationPort using Gemini API / GeminiTranslationProvider.
"""

from __future__ import annotations

import logging
from typing import List

from src.core.contracts import TranslationPort
from src.domain.entities.segment import Segment

logger = logging.getLogger(__name__)


class GeminiTranslationAdapter(TranslationPort):
    def __init__(self, api_key: str | None = None, model: str = "gemini-3.5-flash-lite"):
        from src.infrastructure.providers.translation.gemini_translation_provider import GeminiTranslationProvider

        self.provider = GeminiTranslationProvider(model=model)

    def translate_texts(
        self, texts: List[str], target_language: str = "vi", source_language: str = "auto"
    ) -> List[str]:
        if not texts:
            return []

        # Wrap texts as Segment objects for GeminiTranslationProvider
        segs = [Segment(id=f"t_{i}", start=0.0, end=1.0, text=t) for i, t in enumerate(texts)]
        translated = self.provider.translate(segs, target_language=target_language)
        return [s.translated_text or s.text for s in translated]
