"""MEDIAFLOW_GEMINI_MODEL overrides every Gemini call site (OCR + translation)
without code edits — e.g. switching to a higher-quota Gemma model."""

from src.infrastructure.ocr.gemini_ocr import GeminiOCRAdapter
from src.infrastructure.translation.gemini_translation import GeminiTranslationAdapter


def test_default_models_unchanged_without_env(monkeypatch):
    monkeypatch.delenv("MEDIAFLOW_GEMINI_MODEL", raising=False)
    assert GeminiOCRAdapter(api_key="k").model == "gemini-3.5-flash-lite"
    assert GeminiTranslationAdapter(model="gemini-2.5-flash").model == "gemini-2.5-flash"


def test_env_overrides_all_gemini_call_sites(monkeypatch):
    monkeypatch.setenv("MEDIAFLOW_GEMINI_MODEL", "gemma-4-31b-it")
    assert GeminiOCRAdapter(api_key="k").model == "gemma-4-31b-it"
    # Env wins even where call sites pass an explicit model.
    assert GeminiTranslationAdapter(model="gemini-2.5-flash").model == "gemma-4-31b-it"