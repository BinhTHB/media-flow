from typing import List
from src.application.ports.speech_provider import SpeechProvider
from src.domain.entities.segment import Segment

class TranscribeSpeechUseCase:
    def __init__(self, speech_provider: SpeechProvider):
        self.speech_provider = speech_provider

    def execute(self, audio_path: str, source_language: str) -> List[Segment]:
        return self.speech_provider.transcribe(audio_path, source_language)
