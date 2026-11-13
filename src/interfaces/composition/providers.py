"""Provider wiring shared by local CLI and web entry points."""

from __future__ import annotations

import os

from src.features.translation.service import TranslationService
from src.infrastructure.translation.gemini_translation import GeminiTranslationAdapter


def has_openrouter_keys() -> bool:
    from src.infrastructure.translation.openrouter_translation import configured_openrouter_keys

    return bool(configured_openrouter_keys())


def has_elevenlabs_keys() -> bool:
    from src.infrastructure.elevenlabs_dubbing import configured_elevenlabs_keys

    return bool(configured_elevenlabs_keys())


def build_translation_service(purpose: str, provider_choice: str) -> TranslationService:
    from src.infrastructure.translation.openrouter_translation import OpenRouterTranslationAdapter
    from src.infrastructure.translation.router import FallbackTranslationAdapter

    if provider_choice in {"mock", "auto"}:
        provider_choice = "auto"

    providers = []
    if purpose == "subtitle" and provider_choice == "auto":
        if os.getenv("GEMINI_API_KEY"):
            providers.append(("gemini", GeminiTranslationAdapter(model="gemini-3.5-flash-lite", purpose=purpose)))
        if has_openrouter_keys():
            providers.append(("openrouter", OpenRouterTranslationAdapter(purpose=purpose)))
    else:
        if provider_choice in {"auto", "openrouter"} and has_openrouter_keys():
            providers.append(("openrouter", OpenRouterTranslationAdapter(purpose=purpose)))
        if provider_choice in {"auto", "gemini"} and os.getenv("GEMINI_API_KEY"):
            providers.append(("gemini", GeminiTranslationAdapter(model="gemini-3.5-flash-lite", purpose=purpose)))
    if provider_choice == "openrouter" and not has_openrouter_keys():
        raise RuntimeError("OPENROUTER_API_KEY(S) is required for translation provider openrouter.")
    if provider_choice == "gemini" and not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is required for translation provider gemini.")
    if not providers:
        raise RuntimeError("No translation provider is configured; set OPENROUTER_API_KEY(S) or GEMINI_API_KEY.")
    return TranslationService(FallbackTranslationAdapter(providers))


def build_asr_provider(provider_choice: str):
    from src.infrastructure.speech.gemini_audio import GeminiAudioASRAdapter
    from src.infrastructure.speech.groq_whisper import GroqWhisperASRAdapter
    from src.infrastructure.speech.router import FallbackASRAdapter

    if provider_choice == "mock":
        provider_choice = "auto"

    if provider_choice == "groq":
        return GroqWhisperASRAdapter(model="whisper-large-v3", timeout_seconds=45)
    if provider_choice == "gemini":
        return GeminiAudioASRAdapter()
    providers = []
    if os.getenv("GROQ_API_KEY"):
        providers.append(("groq-whisper-large-v3", GroqWhisperASRAdapter(model="whisper-large-v3", timeout_seconds=45)))
    if os.getenv("GEMINI_API_KEY"):
        providers.append(("gemini-audio", GeminiAudioASRAdapter()))
    if not providers:
        raise RuntimeError("No ASR provider is configured; set GROQ_API_KEY or GEMINI_API_KEY.")
    return providers[0][1] if len(providers) == 1 else FallbackASRAdapter(providers)


def build_asr_corrector(provider_choice: str, asr_provider_choice: str):
    if provider_choice == "off":
        return None
    if provider_choice not in {"auto", "openrouter"}:
        return None
    # providers.yaml still labels speech as "mock". The ASR factory treats that
    # label as auto and calls Groq, so the repair path must do the same.
    effective_asr = "auto" if asr_provider_choice == "mock" else asr_provider_choice
    primary_is_groq = effective_asr == "groq" or (
        effective_asr == "auto" and bool(os.getenv("GROQ_API_KEY"))
    )
    secondary_provider = None
    if primary_is_groq and os.getenv("GEMINI_API_KEY"):
        from src.infrastructure.speech.gemini_audio import GeminiAudioASRAdapter

        secondary_provider = GeminiAudioASRAdapter()
    openrouter_ready = has_openrouter_keys()
    if provider_choice == "openrouter" and not openrouter_ready:
        raise RuntimeError("OpenRouter key pool is required for ASR correction.")
    if not openrouter_ready and secondary_provider is None:
        return None
    from src.infrastructure.speech.openrouter_corrector import OpenRouterASRCorrector

    return OpenRouterASRCorrector(
        api_keys=None if openrouter_ready else [],
        secondary_provider=secondary_provider,
    )


def build_spoken_rewriter(provider_choice: str):
    if provider_choice == "off":
        return None
    if provider_choice in {"auto", "openrouter"} and has_openrouter_keys():
        from src.infrastructure.translation.duration_rewriter import OpenRouterDurationRewriter

        return OpenRouterDurationRewriter()
    if provider_choice == "openrouter":
        raise RuntimeError("OpenRouter key pool is required for duration rewrite.")
    return None


def build_ocr_provider(provider_choice: str = "auto"):
    """Wire OCRPort to local RapidOCR only.

    Gemini is not an OCR provider. Translation still uses its own service.
    """
    del provider_choice
    from src.infrastructure.ocr.rapid_ocr import RapidOCRAdapter, rapidocr_available

    if not rapidocr_available():
        raise RuntimeError("RapidOCR is not installed. pip install rapidocr onnxruntime")
    return RapidOCRAdapter()


def build_tts_provider(provider_choice: str):
    from src.infrastructure.tts.edge_tts import EdgeTTSAdapter
    from src.infrastructure.tts.elevenlabs_tts import ElevenLabsTTSAdapter
    from src.infrastructure.tts.gemini_flash_live import GeminiFlashLiveTTSAdapter
    from src.infrastructure.tts.openrouter_tts import OpenRouterTTSAdapter
    from src.infrastructure.tts.router import FallbackTTSAdapter

    if provider_choice == "mock":
        provider_choice = "auto"

    if provider_choice == "elevenlabs":
        if not has_elevenlabs_keys():
            raise RuntimeError("ELEVENLABS_API_KEY(S) is required for TTS provider elevenlabs.")
        return ElevenLabsTTSAdapter()
    if provider_choice == "fish":
        if not has_openrouter_keys():
            raise RuntimeError("OpenRouter key pool is required for TTS provider fish.")
        return OpenRouterTTSAdapter()
    if provider_choice == "gemini-live":
        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY is required for TTS provider gemini-live.")
        return GeminiFlashLiveTTSAdapter()
    if provider_choice == "edge":
        return EdgeTTSAdapter()

    providers = []
    if has_elevenlabs_keys():
        providers.append(("elevenlabs-v3", ElevenLabsTTSAdapter()))
    if has_openrouter_keys():
        providers.append(("openrouter-fish-s2.1-pro-free", OpenRouterTTSAdapter()))
    providers.append(("edge-tts", EdgeTTSAdapter()))
    return FallbackTTSAdapter(providers)
