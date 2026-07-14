from typing import List
from src.application.ports.frame_extraction_provider import FrameExtractionProvider
from src.domain.entities.frame import Frame

class ExtractFramesUseCase:
    def __init__(self, frame_extraction_provider: FrameExtractionProvider):
        self.frame_extraction_provider = frame_extraction_provider

    def execute(self, video_path: str, interval: float, output_dir: str) -> List[Frame]:
        return self.frame_extraction_provider.extract_frames(video_path, interval, output_dir)
