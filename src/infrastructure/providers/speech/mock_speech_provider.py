from typing import List
from src.application.ports.speech_provider import SpeechProvider
from src.domain.entities.segment import Segment

class MockSpeechProvider(SpeechProvider):
    def transcribe(self, audio_path: str, source_language: str) -> List[Segment]:
        if source_language == "zh":
            return [
                Segment(
                    id="speech_1",
                    start=0.0,
                    end=3.0,
                    text="大家好，欢迎来到我的频道。",
                    translated_text="",
                    source="speech",
                    confidence=0.98,
                ),
                Segment(
                    id="speech_2",
                    start=3.5,
                    end=6.5,
                    text="今天我们来聊聊人工智能。",
                    translated_text="",
                    source="speech",
                    confidence=0.95,
                ),
            ]
        else:
            return [
                Segment(
                    id="speech_1",
                    start=0.0,
                    end=3.0,
                    text="Hello everyone, welcome to my channel.",
                    translated_text="",
                    source="speech",
                    confidence=0.98,
                ),
                Segment(
                    id="speech_2",
                    start=3.5,
                    end=6.5,
                    text="Today we are going to talk about AI.",
                    translated_text="",
                    source="speech",
                    confidence=0.95,
                ),
            ]
