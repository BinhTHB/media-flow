"""OCR failover: local crops first, Gemini only for unreadables."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from src.core.contracts import OCRPort

logger = logging.getLogger(__name__)


class FallbackOCRAdapter(OCRPort):
    def __init__(
        self,
        primary: OCRPort,
        fallback: Optional[OCRPort] = None,
        *,
        max_fallback: int = 8,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.max_fallback = max(0, int(max_fallback))
        primary_model = str(getattr(primary, "model", type(primary).__name__))
        fallback_model = str(getattr(fallback, "model", "none") if fallback is not None else "none")
        self.model = f"{primary_model}+{fallback_model}"

    def read_crops(self, crop_images: List[Any]) -> List[str]:
        if not crop_images:
            return []
        texts = list(self.primary.read_crops(crop_images))
        if len(texts) != len(crop_images):
            raise RuntimeError("OCR primary returned a mismatched batch.")
        if self.fallback is None:
            return texts
        empty_idx = [index for index, text in enumerate(texts) if not str(text or "").strip()]
        if not empty_idx:
            return texts
        if self.max_fallback and len(empty_idx) > self.max_fallback:
            empty_idx = empty_idx[: self.max_fallback]
        filled = self.fallback.read_crops([crop_images[index] for index in empty_idx])
        recovered = 0
        for index, text in zip(empty_idx, filled):
            if str(text or "").strip():
                texts[index] = text
                recovered += 1
        logger.info(
            "OCR: local read %d/%d crops; Gemini filled %d of %d empty.",
            len(crop_images) - len(empty_idx),
            len(crop_images),
            recovered,
            len(empty_idx),
        )
        return texts
