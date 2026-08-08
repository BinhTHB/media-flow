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

        prompt_text = (
            "Transcribe each on-screen subtitle image crop into its exact visible source characters.\n"
            "Rules:\n"
            "1. Transcribe ONLY the characters physically visible in the crop image.\n"
            "2. Do NOT extrapolate missing words, complete sentences, or guess unreadable text.\n"
            "3. Do NOT repeat words or characters unless they are physically repeated multiple times in the image.\n"
            "4. Do NOT translate, paraphrase, or rewrite.\n"
            "5. If text is unreadable or absent, return an empty string \"\".\n"
            "6. Return ONLY a JSON object mapping each item_id to its exact transcribed text:\n"
            '{\n  "item_0": "exact_visible_text_0",\n  "item_1": "exact_visible_text_1"\n}'
        )

        parts: list[dict[str, Any]] = [{"text": prompt_text}]

        valid_item_ids = []
        for idx, crop in enumerate(crop_images):
            if crop is None or crop.size == 0:
                continue
            ok, encoded = cv2.imencode(".png", crop)
            if not ok:
                continue
            item_id = f"item_{idx}"
            parts.extend([
                {"text": f"item_id={item_id}"},
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
                    }
                },
            ])
            valid_item_ids.append(item_id)

        if not valid_item_ids:
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

            if isinstance(payload, dict):
                for idx, item_id in enumerate(valid_item_ids):
                    val = payload.get(item_id)
                    if isinstance(val, str):
                        results[idx] = val.strip()
            elif isinstance(payload, list):
                for idx_pos, val in enumerate(payload):
                    if idx_pos < len(valid_item_ids):
                        if isinstance(val, dict) and "text" in val:
                            results[idx_pos] = str(val["text"]).strip()
                        elif isinstance(val, str):
                            results[idx_pos] = val.strip()

            return results
        except Exception as exc:
            logger.warning("GeminiOCRAdapter.read_crops failed: %s", exc)
            return ["" for _ in crop_images]
