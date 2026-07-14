import os
import time
from typing import List
from src.application.ports.translation_provider import TranslationProvider
from src.domain.entities.segment import Segment
from src.shared.exceptions import ProviderError

class OpenAITranslationProvider(TranslationProvider):
    def __init__(self, model: str = "gpt-4o-mini", max_retries: int = 3):
        self.model = model
        self.max_retries = max_retries
        self._client = None

    def _get_client(self):
        if self._client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ProviderError("OPENAI_API_KEY environment variable is not set.")
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key)
            except ImportError as e:
                raise ProviderError(
                    "openai package is not installed. Install it with: pip install openai"
                ) from e
        return self._client

    def translate(self, segments: List[Segment], target_language: str) -> List[Segment]:
        if not segments:
            return []
            
        client = self._get_client()
        texts_to_translate = [seg.text for seg in segments]
        translated_texts = self._batch_translate(client, texts_to_translate, target_language)
        
        for seg, translated_text in zip(segments, translated_texts):
            seg.translated_text = translated_text
        
        return segments

    def _batch_translate(self, client, texts: List[str], target_lang: str) -> List[str]:
        if not texts:
            return []
        
        lines = [f"{i}: {text}" for i, text in enumerate(texts)]
        prompt = "\n".join(lines)
        
        system_msg = (
            f"You are a professional translator. Translate each of the following lines to {target_lang}. "
            "Preserve line numbers in the format 'i: translated_text'. Do not add any explanations or extra text."
        )
        
        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                )
                content = response.choices[0].message.content.strip()
                
                result = {}
                for line in content.split("\n"):
                    line = line.strip()
                    if ":" in line:
                        idx_str, _, text = line.partition(":")
                        idx_str = idx_str.strip()
                        if idx_str.isdigit():
                            result[int(idx_str)] = text.strip()
                
                translated = [result.get(i, texts[i]) for i in range(len(texts))]
                return translated
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise ProviderError(f"OpenAI translation failed after {self.max_retries} retries: {e}") from e
