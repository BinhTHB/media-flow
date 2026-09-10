"""Failover router for ASR providers."""

from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

from src.core.contracts import ASRPort
from src.core.models.speech_segment import SpeechSegment

logger = logging.getLogger(__name__)


class FallbackASRAdapter(ASRPort):
    def __init__(self, providers: Sequence[Tuple[str, ASRPort]]) -> None:
        self.providers = [(name, provider) for name, provider in providers if provider is not None]

    def transcribe(self, audio_path: str, language: str = "auto") -> List[SpeechSegment]:
        if not self.providers:
            raise RuntimeError("No ASR provider is configured.")
        errors = []
        for name, provider in self.providers:
            try:
                result = provider.transcribe(audio_path=audio_path, language=language)
                if result:
                    return result
                raise RuntimeError("provider returned no speech segments")
            except Exception as exc:
                logger.warning("ASR provider %s failed; trying fallback: %s", name, exc)
                errors.append(f"{name}: {str(exc)[:240]}")
        raise RuntimeError("All ASR providers failed. " + " | ".join(errors))
