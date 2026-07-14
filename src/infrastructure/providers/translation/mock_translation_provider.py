from typing import List
from src.application.ports.translation_provider import TranslationProvider
from src.domain.entities.segment import Segment

class MockTranslationProvider(TranslationProvider):
    def translate(self, segments: List[Segment], target_language: str) -> List[Segment]:
        translations = {
            "大家好，欢迎来到我的频道。": "Chào mọi người, chào mừng đến với kênh của tôi.",
            "今天我们来聊聊人工智能。": "Hôm nay chúng ta sẽ nói về trí tuệ nhân tạo.",
            "Hello everyone, welcome to my channel.": "Chào mọi người, chào mừng đến với kênh của tôi.",
            "Today we are going to talk about AI.": "Hôm nay chúng ta sẽ nói về trí tuệ nhân tạo.",
            "AI Video Translator": "Trình dịch Video AI",
        }
        for segment in segments:
            segment.translated_text = translations.get(segment.text, f"[Dịch sang {target_language}]: {segment.text}")
        return segments
