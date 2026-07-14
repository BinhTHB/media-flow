from abc import ABC, abstractmethod
from typing import List
from src.domain.entities.frame import Frame

class FrameExtractionProvider(ABC):
    @abstractmethod
    def extract_frames(self, video_path: str, interval: float, output_dir: str) -> List[Frame]:
        pass
