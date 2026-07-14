from src.application.services.subtitle_segmentation_service import SubtitleSegmentationService
from src.domain.entities.segment import Segment

class SubtitleSegmentationUseCase:
    def __init__(self, service: SubtitleSegmentationService):
        self.service = service

    def execute(self, segments: list[Segment], is_translated: bool = False) -> list[Segment]:
        return self.service.segment(segments, is_translated=is_translated)
