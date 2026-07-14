from typing import List, Optional

from src.application.ports.subtitle_placement_debug_provider import (
    SubtitlePlacementDebugProvider,
)
from src.domain.entities.frame import Frame
from src.domain.entities.subtitle_placement import SubtitlePlacement
from src.domain.entities.subtitle_region import SubtitleRegion


class GenerateSubtitlePlacementDebugUseCase:
    def __init__(self, provider: SubtitlePlacementDebugProvider) -> None:
        self.provider = provider

    def execute(
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
        self.provider.generate(
            frames,
            regions,
            placements,
            before_path,
            overlay_path,
            after_path,
            regions_path,
            placements_path,
            rendered_video_path,
        )
