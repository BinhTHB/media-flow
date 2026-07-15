import time
from typing import List, Optional

from src.application.ports.translation_provider import TranslationProvider
from src.domain.entities.segment import Segment
from src.infrastructure.gemini.contracts import GeminiRequest
from src.infrastructure.gemini.exceptions import GeminiAuthError, GeminiError
from src.infrastructure.gemini.runtime import GeminiRuntime
from src.shared.exceptions import ProviderError



class GeminiTranslationProvider(TranslationProvider):
    def __init__(
        self,
        runtime: Optional[GeminiRuntime] = None,
        model: str = "gemini-3.1-flash-lite",
        max_retries: int = 3,
    ):
        self._runtime = runtime or GeminiRuntime.create()
        self.model = model
        self.max_retries = max_retries

    def translate(self, segments: List[Segment], target_language: str) -> List[Segment]:
        if not segments:
            return []

        chunk_size = 15
        translated_texts: List[str] = []
        start_time = time.monotonic()
        for i in range(0, len(segments), chunk_size):
            chunk = segments[i : i + chunk_size]
            chunk_texts = [seg.text for seg in chunk]
            chunk_translations = self._translate_chunk(chunk_texts, target_language)
            translated_texts.extend(chunk_translations)
            if i + chunk_size < len(segments):
                # Minimum 3s gap between chunks (max 20 RPM)
                elapsed = time.monotonic() - start_time
                gap = 3.0 - elapsed
                if gap > 0:
                    time.sleep(gap)
                start_time = time.monotonic()

        for segment, translated_text in zip(segments, translated_texts):
            segment.translated_text = translated_text
        return segments

    def _translate_chunk(self, texts: List[str], target_language: str) -> List[str]:
        numbered_lines = "\n".join(f"{index}: {text}" for index, text in enumerate(texts))
        prompt = (
            f"Translate each numbered line into {target_language}. "
            "Preserve every line number using the exact format 'i: translated_text'. "
            "Return only translated numbered lines with no explanations.\n\n"
            f"{numbered_lines}"
        )

        for attempt in range(self.max_retries):
            try:
                client = self._runtime.get_client(model=self.model)
                response = client.generate_content(
                    GeminiRequest(
                        contents=[{"parts": [{"text": prompt}]}],
                        generation_config={"temperature": 0.3},
                    )
                )
                return self._parse_numbered_lines(response.text.strip(), texts)
            except GeminiAuthError as exc:
                raise ProviderError(
                    "GEMINI_API_KEY environment variable is not set."
                ) from exc
            except GeminiError as exc:
                if attempt == self.max_retries - 1:
                    raise ProviderError(
                        f"Gemini translation failed after {self.max_retries} retries: {exc}"
                    ) from exc
                time.sleep(2 ** attempt)

        return texts

    @staticmethod
    def _parse_numbered_lines(content: str, original_texts: List[str]) -> List[str]:
        translated = {}
        for line in content.splitlines():
            index_text, separator, text = line.partition(":")
            if separator and index_text.strip().isdigit():
                translated[int(index_text.strip())] = text.strip()
        return [translated.get(index, original_texts[index]) for index in range(len(original_texts))]
