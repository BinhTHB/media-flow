from typing import List
from src.domain.entities.segment import Segment

class MergeSegmentsUseCase:
    def execute(self, speech_segments: List[Segment], ocr_segments: List[Segment]) -> List[Segment]:
        merged = []
        merged.extend(speech_segments)
        merged.extend(ocr_segments)
        merged.sort(key=lambda s: s.start)
        return merged
