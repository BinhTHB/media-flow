from abc import ABC, abstractmethod
from typing import List
from src.domain.entities.frame import Frame
from src.domain.entities.segment import Segment

class OCRProvider(ABC):
    @abstractmethod
    def detect_text(self, frames: List[Frame]) -> List[Segment]:
        pass
