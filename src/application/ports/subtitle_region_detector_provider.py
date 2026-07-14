from abc import ABC, abstractmethod
from typing import List

from src.domain.entities.frame import Frame
from src.domain.entities.subtitle_region import SubtitleRegion


class SubtitleRegionDetectorProvider(ABC):
    @abstractmethod
    def detect_regions(self, frames: List[Frame]) -> List[SubtitleRegion]:
        pass
