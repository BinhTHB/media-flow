from typing import List
from src.application.ports.translation_provider import TranslationProvider
from src.domain.entities.segment import Segment

class TranslateSegmentsUseCase:
    def __init__(self, translation_provider: TranslationProvider):
        self.translation_provider = translation_provider

    def execute(self, segments: List[Segment], target_language: str) -> List[Segment]:
        return self.translation_provider.translate(segments, target_language)
