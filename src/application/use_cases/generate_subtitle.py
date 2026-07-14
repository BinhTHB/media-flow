from typing import List, Optional
from src.application.ports.subtitle_provider import SubtitleProvider
from src.domain.entities.segment import Segment
from src.domain.entities.subtitle_placement import SubtitlePlacement
from src.domain.entities.subtitle_style import SubtitleStyle
from src.shared.exceptions import UseCaseError


class GenerateSubtitleUseCase:
    def __init__(self, subtitle_provider: SubtitleProvider):
        self.subtitle_provider = subtitle_provider

    def execute(
        self,
        segments: List[Segment],
        subtitle_format: str = "srt",
        style: Optional[SubtitleStyle] = None,
        placements: Optional[List[SubtitlePlacement]] = None,
        play_res_x: int = 1920,
        play_res_y: int = 1080,
    ) -> str:
        if subtitle_format == "srt":
            return self.subtitle_provider.generate_srt(segments)
        elif subtitle_format == "ass":
            return self.subtitle_provider.generate_ass(
                segments, style or SubtitleStyle(), placements, play_res_x, play_res_y
            )
        else:
            raise UseCaseError(f"Unsupported subtitle format: {subtitle_format}")
