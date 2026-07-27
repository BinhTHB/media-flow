from typing import List, Optional
from src.application.ports.subtitle_provider import SubtitleProvider
from src.domain.entities.segment import Segment
from src.domain.entities.subtitle_placement import SubtitlePlacement
from src.domain.entities.subtitle_region import SubtitleRegion
from src.domain.entities.subtitle_style import SubtitleStyle
from src.application.services.mask_authority import DecisionStore
from src.application.services.ass_generator import ASSGenerator

class MockSubtitleProvider(SubtitleProvider):
    def generate_srt(self, segments: List[Segment]) -> str:
        from src.domain.entities.subtitle_item import SubtitleItem
        srt_parts = []
        for idx, segment in enumerate(segments, start=1):
            text = segment.translated_text if segment.translated_text else segment.text
            item = SubtitleItem(
                index=idx,
                start=segment.start,
                end=segment.end,
                text=text,
            )
            srt_parts.append(item.to_srt_format())
        return "\n".join(srt_parts)

    def generate_ass(
        self,
        segments: List[Segment],
        style: SubtitleStyle,
        placements: Optional[List[SubtitlePlacement]] = None,
        play_res_x: int = 1920,
        play_res_y: int = 1080,
        mask_regions: Optional[List[SubtitleRegion]] = None,
        decision_store: Optional[DecisionStore] = None,
    ) -> str:
        return ASSGenerator.generate(
            segments, style, placements, play_res_x, play_res_y, mask_regions,
            decision_store=decision_store,
        )
