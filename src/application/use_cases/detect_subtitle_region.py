from typing import List

from src.application.ports.subtitle_region_detector_provider import (
    SubtitleRegionDetectorProvider,
)
from src.domain.entities.frame import Frame
from src.domain.entities.subtitle_region import SubtitleRegion


class DetectSubtitleRegionUseCase:
    def __init__(self, provider: SubtitleRegionDetectorProvider) -> None:
        self.provider = provider

    def execute(self, frames: List[Frame]) -> List[SubtitleRegion]:
        return self.provider.detect_regions(frames)
