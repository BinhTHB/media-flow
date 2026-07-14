from typing import List
from src.application.ports.ocr_provider import OCRProvider
from src.domain.entities.frame import Frame
from src.domain.entities.segment import Segment

class DetectOnscreenTextUseCase:
    def __init__(self, ocr_provider: OCRProvider):
        self.ocr_provider = ocr_provider

    def execute(self, frames: List[Frame]) -> List[Segment]:
        return self.ocr_provider.detect_text(frames)
