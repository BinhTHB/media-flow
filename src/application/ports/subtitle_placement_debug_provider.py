from abc import ABC, abstractmethod
from typing import List, Optional

from src.domain.entities.frame import Frame
from src.domain.entities.subtitle_placement import SubtitlePlacement
from src.domain.entities.subtitle_region import SubtitleRegion


class SubtitlePlacementDebugProvider(ABC):
    @abstractmethod
    def generate(
        self,
        frames: List[Frame],
        regions: List[SubtitleRegion],
        placements: List[SubtitlePlacement],
        before_path: str,
        overlay_path: str,
        after_path: str,
        regions_path: str,
        placements_path: str,
        rendered_video_path: Optional[str] = None,
    ) -> None:
        pass
