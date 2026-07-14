from typing import List

from src.application.ports.subtitle_region_detector_provider import (
    SubtitleRegionDetectorProvider,
)
from src.domain.entities.frame import Frame
from src.domain.entities.subtitle_region import SubtitleRegion


class MockSubtitleRegionDetectorProvider(SubtitleRegionDetectorProvider):
    def detect_regions(self, frames: List[Frame]) -> List[SubtitleRegion]:
        return [
            SubtitleRegion(
                start_time=frame.timestamp,
                end_time=frame.timestamp + 2.0,
                x=0.1,
                y=0.8,
                width=0.8,
                height=0.08,
                confidence=0.95,
            )
            for frame in frames
        ]
