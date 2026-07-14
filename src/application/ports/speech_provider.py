from abc import ABC, abstractmethod
from typing import List
from src.domain.entities.segment import Segment

class SpeechProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str, source_language: str) -> List[Segment]:
        pass
