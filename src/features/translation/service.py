"""
Shared Translation Service

Provides text translation capability to both AudioPipeline and SubtitleProcessor.
Does NOT read images, read audio, create bboxes, or decide rendering.
"""

from __future__ import annotations

from typing import List

from src.core.contracts import TranslationPort


class TranslationService:
    def __init__(self, provider: TranslationPort):
        self.provider = provider

    def translate_texts(
        self, texts: List[str], target_language: str = "vi", source_language: str = "auto"
    ) -> List[str]:
        if not texts:
            return []
        return self.provider.translate_texts(
            texts=texts, target_language=target_language, source_language=source_language
        )
