import json
import os
import time
from typing import List
from urllib import error, request

from src.application.ports.translation_provider import TranslationProvider
from src.domain.entities.segment import Segment
from src.shared.exceptions import ProviderError


class GeminiTranslationProvider(TranslationProvider):
    def __init__(
        self,
        model: str = "gemini-3.1-flash-lite",
        max_retries: int = 3,
        api_key: str | None = None,
    ):
        self.model = model
        self.max_retries = max_retries
        self.api_key = api_key

    def translate(self, segments: List[Segment], target_language: str) -> List[Segment]:
        if not segments:
            return []

        chunk_size = 15
        api_key = self.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ProviderError("GEMINI_API_KEY environment variable is not set.")

        translated_texts = []
        start_time = time.monotonic()
        for i in range(0, len(segments), chunk_size):
            chunk = segments[i : i + chunk_size]
            chunk_texts = [seg.text for seg in chunk]
            chunk_translations = self._translate_chunk(chunk_texts, target_language, api_key)
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

    def _translate_chunk(self, texts: List[str], target_language: str, api_key: str) -> List[str]:
        numbered_lines = "\n".join(f"{index}: {text}" for index, text in enumerate(texts))
        prompt = (
            f"Translate each numbered line into {target_language}. "
            "Preserve every line number using the exact format 'i: translated_text'. "
            "Return only translated numbered lines with no explanations.\n\n"
            f"{numbered_lines}"
        )
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3},
        }).encode("utf-8")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={api_key}"

        for attempt in range(self.max_retries):
            try:
                http_request = request.Request(
                    url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with request.urlopen(http_request, timeout=60) as response:
                    body = json.loads(response.read().decode("utf-8"))

                if "error" in body:
                    raise ProviderError(f"Gemini API error: {body['error']['message']}")

                content = body["candidates"][0]["content"]["parts"][0]["text"].strip()
                parsed = self._parse_numbered_lines(content, texts)
                return parsed
            except (error.URLError, error.HTTPError, KeyError, IndexError, json.JSONDecodeError) as exc:
                wait_time = 2 ** attempt
                if isinstance(exc, error.HTTPError) and exc.code == 429:
                    # Try to respect Retry-After header
                    retry_after = exc.headers.get("Retry-After")
                    wait_time = max(float(retry_after), 5 + attempt * 5) if retry_after else 5 + attempt * 5
                if attempt == self.max_retries - 1:
                    raise ProviderError(
                        f"Gemini translation failed after {self.max_retries} retries: {exc}"
                    ) from exc
                time.sleep(wait_time)

        return texts

    @staticmethod
    def _parse_numbered_lines(content: str, original_texts: List[str]) -> List[str]:
        translated = {}
        for line in content.splitlines():
            index_text, separator, text = line.partition(":")
            if separator and index_text.strip().isdigit():
                translated[int(index_text.strip())] = text.strip()
        return [translated.get(index, original_texts[index]) for index in range(len(original_texts))]
