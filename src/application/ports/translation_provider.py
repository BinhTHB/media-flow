from abc import ABC, abstractmethod
from typing import List
from src.domain.entities.segment import Segment

class TranslationProvider(ABC):
    @abstractmethod
    def translate(self, segments: List[Segment], target_language: str) -> List[Segment]:
        pass
