from pathlib import Path
from typing import List, Optional

from src.application.ports.ocr_provider import OCRProvider, VideoSubtitleOCRProvider
from src.domain.entities.frame import Frame
from src.domain.entities.segment import Segment

class DetectOnscreenTextUseCase:
    def __init__(self, ocr_provider: OCRProvider | VideoSubtitleOCRProvider):
        self.ocr_provider = ocr_provider

    def execute(
        self,
        frames: Optional[List[Frame]] = None,
        *,
        video_path: str | Path | None = None,
        source_language: str | None = None,
    ) -> List[Segment]:
        if isinstance(self.ocr_provider, VideoSubtitleOCRProvider):
            if video_path is None:
                raise ValueError("Direct-video OCR requires video_path.")
            return self.ocr_provider.extract_subtitles(
                Path(video_path), source_language
            )
        return self.ocr_provider.detect_text(frames)
