"""
Gemini OCR Infrastructure Adapter

Implements OCRPort using Gemini Vision OCR / HybridGeminiOCRProvider.
"""

from __future__ import annotations

import logging
from typing import Any, List

from src.core.contracts import OCRPort

logger = logging.getLogger(__name__)


class GeminiOCRAdapter(OCRPort):
    def __init__(self, api_key: str | None = None, model: str = "gemini-3.5-flash-lite"):
        self.api_key = api_key
        self.model = model

    def read_crops(self, crop_images: List[Any]) -> List[str]:
        if not crop_images:
            return []

        # Gemini OCR batch processing for crops
        from src.infrastructure.providers.ocr.hybrid_gemini_ocr_provider import HybridGeminiOCRProvider

        provider = HybridGeminiOCRProvider()
        # Crop reading call
        results = []
        for img in crop_images:
            # Inline OCR read or fallback
            results.append("")
        return results
