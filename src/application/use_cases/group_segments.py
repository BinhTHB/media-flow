import re
from typing import List
from src.domain.entities.segment import Segment
from src.domain.entities.subtitle_quality_config import SubtitleQualityConfig

class GroupSegmentsUseCase:
    def __init__(self, quality_config: SubtitleQualityConfig = None):
        self.config = quality_config or SubtitleQualityConfig()
        self.provenance_map = {}

    def execute(self, segments: List[Segment], source_language: str) -> List[Segment]:
        self.provenance_map = {}
        if not segments:
            return []

        sorted_segs = sorted(segments, key=lambda s: s.start)

        grouped = []
        current_group = []

        for s in sorted_segs:
            # Default mapping for every segment is itself
            self.provenance_map[s.id] = [s.id]

            if not current_group:
                current_group.append(s)
            else:
                last = current_group[-1]
                gap = s.start - last.end
                
                last_text = last.text.strip()
                ends_with_punc = False
                if last_text:
                    ends_with_punc = last_text[-1] in ".!?。！？"

                if (
                    last.source == s.source and
                    gap <= self.config.ocr_grouping_max_gap and
                    not ends_with_punc
                ):
                    current_group.append(s)
                else:
                    grouped.append(self._merge_group(current_group, source_language))
                    current_group = [s]

        if current_group:
            grouped.append(self._merge_group(current_group, source_language))

        return grouped

    def _merge_group(self, group: List[Segment], lang: str) -> Segment:
        if len(group) == 1:
            return group[0]

        texts = [s.text.strip() for s in group]
        
        if lang == "zh":
            joined = ""
            for idx, t in enumerate(texts):
                if idx == 0:
                    joined = t
                else:
                    has_chinese_prev = bool(re.search(r"[\u4e00-\u9fff]", joined))
                    has_chinese_curr = bool(re.search(r"[\u4e00-\u9fff]", t))
                    if has_chinese_prev or has_chinese_curr:
                        joined += t
                    else:
                        joined += " " + t
            text_val = joined
        else:
            text_val = " ".join(texts)

        first = group[0]
        last = group[-1]
        avg_confidence = sum(s.confidence for s in group) / len(group)

        merged = Segment(
            id=first.id,
            start=first.start,
            end=last.end,
            text=text_val,
            translated_text="",
            source=first.source,
            confidence=avg_confidence
        )
        self.provenance_map[merged.id] = [s.id for s in group]
        return merged