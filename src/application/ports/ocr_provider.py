from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from src.domain.entities.frame import Frame
from src.domain.entities.segment import Segment

class OCRProvider(ABC):
    @abstractmethod
    def detect_text(self, frames: List[Frame]) -> List[Segment]:
        pass


class VideoSubtitleOCRProvider(ABC):
    """Port for OCR providers that operate on the original video stream."""

    @abstractmethod
    def extract_subtitles(
        self, video_path: Path, source_language: str | None
    ) -> List[Segment]:
        pass
