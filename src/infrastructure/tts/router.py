"""Failover router for text-to-speech providers."""

from __future__ import annotations

import logging
from typing import Sequence, Tuple

from src.core.contracts import TTSPort

logger = logging.getLogger(__name__)


class FallbackTTSAdapter(TTSPort):
    def __init__(self, providers: Sequence[Tuple[str, TTSPort]]) -> None:
        self.providers = [(name, provider) for name, provider in providers if provider is not None]
        self.last_provider_name = ""

    def synthesize(self, text: str, output_path: str, voice: str = "vi-VN-HoaiMyNeural") -> str:
        if not self.providers:
            raise RuntimeError("No TTS provider is configured.")
        errors = []
        providers = self.providers
        voice_value = (voice or "").strip()
        if voice_value.startswith("elevenlabs:"):
            providers = sorted(
                self.providers,
                key=lambda item: 0 if "eleven" in item[0].lower() else 1,
            )
        elif voice_value.startswith("fish:"):
            # Explicit Fish reference selected in UI/CLI: do not waste a request
            # on another provider before reaching Fish.
            providers = sorted(
                self.providers,
                key=lambda item: 0 if "fish" in item[0].lower() else 1,
            )
        elif voice_value.startswith("gemini-live:"):
            providers = sorted(
                self.providers,
                key=lambda item: 0 if "gemini" in item[0].lower() and "live" in item[0].lower() else 1,
            )
        for name, provider in providers:
            try:
                provider_voice = voice
                lowered_name = name.lower()
                # If an explicitly selected provider fails/unavailable, pass a
                # provider-neutral Vietnamese voice to fallbacks instead of leaking
                # an incompatible prefixed id (fish:/elevenlabs:/gemini-live:).
                if voice_value.startswith("elevenlabs:") and "eleven" not in lowered_name:
                    provider_voice = "vi-VN-NamMinhNeural"
                elif voice_value.startswith("fish:") and "fish" not in lowered_name:
                    provider_voice = "vi-VN-NamMinhNeural"
                elif voice_value.startswith("gemini-live:") and not (
                    "gemini" in lowered_name and "live" in lowered_name
                ):
                    provider_voice = "vi-VN-NamMinhNeural"
                result = provider.synthesize(text=text, output_path=output_path, voice=provider_voice)
                self.last_provider_name = name
                return result
            except Exception as exc:
                logger.warning("TTS provider %s failed; trying fallback: %s", name, exc)
                errors.append(f"{name}: {str(exc)[:240]}")
        raise RuntimeError("All TTS providers failed. " + " | ".join(errors))
