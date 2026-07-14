from typing import List
from src.application.ports.ocr_provider import OCRProvider
from src.domain.entities.frame import Frame
from src.domain.entities.segment import Segment

class MockOCRProvider(OCRProvider):
    def detect_text(self, frames) -> List[Segment]:
        segments = []
        for idx, frame in enumerate(frames):
            segments.append(
                Segment(
                    id=f"ocr_{idx}",
                    start=frame.timestamp,
                    end=frame.timestamp + 4.0,
                    text=f"AI Video Translator Frame {idx}",
                    translated_text="",
                    source="ocr",
                    confidence=0.92,
                )
            )
        if not segments:
            segments.append(
                Segment(
                    id="ocr_1",
                    start=1.0,
                    end=5.0,
                    text="AI Video Translator",
                    translated_text="",
                    source="ocr",
                    confidence=0.92,
                )
            )
        return segments
