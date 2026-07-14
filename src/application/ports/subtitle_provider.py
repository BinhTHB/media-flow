from abc import ABC, abstractmethod
from typing import List, Optional
from src.domain.entities.segment import Segment
from src.domain.entities.subtitle_placement import SubtitlePlacement
from src.domain.entities.subtitle_style import SubtitleStyle

class SubtitleProvider(ABC):
    @abstractmethod
    def generate_srt(self, segments: List[Segment]) -> str:
        pass

    @abstractmethod
    def generate_ass(
        self,
        segments: List[Segment],
        style: SubtitleStyle,
        placements: Optional[List[SubtitlePlacement]] = None,
        play_res_x: int = 1920,
        play_res_y: int = 1080,
    ) -> str:
        pass
