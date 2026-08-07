"""
Gemini OCR Infrastructure Adapter

Implements OCRPort using Gemini Vision API.
Transcribes exact source text from cropped subtitle images.
Does NOT translate, invent output, or modify bboxes.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, List

import cv2

from src.core.contracts import OCRPort

logger = logging.getLogger(__name__)


class GeminiOCRAdapter(OCRPort):
    def __init__(self, api_key: str | None = None, model: str = "gemini-3.5-flash-lite"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model

    def read_crops(self, crop_images: List[Any]) -> List[str]:
        if not crop_images or not self.api_key:
            return ["" for _ in crop_images]

        parts: list[dict[str, Any]] = [{
            "text": (
                "Transcribe each on-screen subtitle image crop exactly into its original source characters. "
                "Return only a JSON array with one string per crop image: "
                '["original_text_1", "original_text_2"]. '
                "Do not translate, classify, or explain."
            )
        }]

        valid_indices = []
        for idx, crop in enumerate(crop_images):
            if crop is None or crop.size == 0:
                continue
            ok, encoded = cv2.imencode(".png", crop)
            if not ok:
                continue
            parts.extend([
                {"text": f"crop_index={idx}"},
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
                    }
                },
            ])
            valid_indices.append(idx)

        if not valid_indices:
            return ["" for _ in crop_images]

        try:
            from src.infrastructure.gemini.client import GeminiClient
            from src.infrastructure.gemini.contracts import GeminiRequest
            from src.infrastructure.gemini.structured_output import parse_json

            client = GeminiClient(api_key=self.api_key)
            response = client.generate_content(
                GeminiRequest(
                    contents=[{"parts": parts}],
                    model=self.model,
                    generation_config={"temperature": 0.0},
                    request_id="v2_crop_ocr_batch",
                    source_kind="v2_crop_ocr",
                    api_surface="generate_content",
                )
            )

            payload = parse_json(response.text)
            results = ["" for _ in crop_images]

            if isinstance(payload, list):
                for idx_pos, val in enumerate(payload):
                    if idx_pos < len(valid_indices) and isinstance(val, str):
                        results[valid_indices[idx_pos]] = val.strip()

            return results
        except Exception as exc:
            logger.warning("GeminiOCRAdapter.read_crops failed: %s", exc)
            return ["" for _ in crop_images]
