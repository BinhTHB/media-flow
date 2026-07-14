from typing import List, Optional

from src.application.services.subtitle_placement_service import (
    SubtitlePlacementService,
)
from src.domain.entities.segment import Segment
from src.domain.entities.subtitle_placement import SubtitlePlacement
from src.domain.entities.subtitle_region import SubtitleRegion
from src.domain.entities.subtitle_style import SubtitleStyle


class CalculateSubtitlePlacementUseCase:
    def __init__(self, placement_service: SubtitlePlacementService) -> None:
        self.placement_service = placement_service

    def execute(
        self,
        regions: List[SubtitleRegion],
        subtitles: List[Segment],
        style: Optional[SubtitleStyle] = None,
    ) -> List[SubtitlePlacement]:
        if style is None:
            return self.placement_service.calculate_placements(regions, subtitles)
        return self.placement_service.calculate_placements(regions, subtitles, style)

    def quality_report(
        self,
        regions: List[SubtitleRegion],
        subtitles: List[Segment],
        placements: List[SubtitlePlacement],
        style: SubtitleStyle,
    ) -> dict:
        return self.placement_service.build_quality_report(
            regions, subtitles, placements, style
        )
