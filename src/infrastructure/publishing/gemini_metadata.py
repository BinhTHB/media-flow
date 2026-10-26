"""Metadata generation via the project's shared Gemini runtime.

Asks the model for localized title/description/tags from the filename and
any sidecar hints; on ANY model failure it falls back to filename-derived
metadata — an unattended worker must never die on a metadata hiccup."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.contracts import MetadataGenerationPort
from src.core.models.publication import VideoMetadata, fallback_metadata
from src.infrastructure.gemini.contracts import GeminiRequest
from src.infrastructure.gemini.runtime import GeminiRuntime

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.MULTILINE)


class GeminiMetadataGenerator(MetadataGenerationPort):
    def __init__(self, runtime: Optional[GeminiRuntime] = None, model: Optional[str] = None) -> None:
        self._runtime = runtime or GeminiRuntime()
        self.model = os.getenv("MEDIAFLOW_GEMINI_MODEL") or model

    def generate(self, video_path: str, hints: Optional[Dict[str, Any]] = None) -> VideoMetadata:
        stem = Path(video_path).stem
        hint_text = " ".join(f"{k}: {v}" for k, v in (hints or {}).items() if v)
        prompt = (
            "You write YouTube metadata in Vietnamese. From this video filename"
            f" ('{stem}') and hints ({hint_text or 'none'}), return STRICT JSON only:\n"
            '{"title": "<max 95 chars, catchy, no clickbait>", '
            '"description": "<2-3 sentences>", '
            '"tags": ["<8 lowercase tags>"]}\n'
            "No markdown, no commentary."
        )
        try:
            client = self._runtime.get_client(model=self.model)
            response = client.generate_content(
                GeminiRequest(
                    contents=[{"parts": [{"text": prompt}]}],
                    generation_config={"temperature": 0.4},
                    source_kind="publish_metadata",
                )
            )
            return self._parse(response.text, video_path)
        except Exception as exc:
            logger.warning(
                "Publish: metadata generation failed (%s); using filename fallback for %s.",
                exc, stem,
            )
            return fallback_metadata(video_path)

    @staticmethod
    def _parse(text: str, video_path: str) -> VideoMetadata:
        try:
            cleaned = _FENCE_RE.sub("", text.strip()).strip()
            # Tolerate trailing prose after the JSON object.
            if not cleaned.startswith("{"):
                start = cleaned.find("{")
                end = cleaned.rfind("}")
                cleaned = cleaned[start : end + 1]
            payload = json.loads(cleaned)
            metadata = VideoMetadata(
                title=str(payload.get("title") or "")[:100],
                description=str(payload.get("description") or "")[:5000],
                tags=[str(t)[:100] for t in payload.get("tags") or []][:30],
            )
            if metadata.title.strip():
                return metadata
        except (ValueError, AttributeError) as exc:
            logger.warning("Publish: unparseable metadata response (%s); fallback.", exc)
        return fallback_metadata(video_path)
