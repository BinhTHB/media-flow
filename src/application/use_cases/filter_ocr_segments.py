import re
from typing import List, Tuple
from src.domain.entities.segment import Segment
from src.domain.entities.subtitle_quality_config import SubtitleQualityConfig
from src.application.services.text_similarity import TextSimilarityService

class FilterOCRSegmentsUseCase:
    def __init__(self, config: SubtitleQualityConfig = None):
        self.config = config or SubtitleQualityConfig()

    def execute(self, segments: List[Segment], source_language: str) -> Tuple[List[Segment], dict]:
        ocr_segs = [s for s in segments if s.source == "ocr"]
        non_ocr_segs = [s for s in segments if s.source != "ocr"]

        before_count = len(ocr_segs)
        dropped_confidence = 0
        dropped_length = 0
        dropped_abnormal = 0
        dropped_repetition = 0
        dropped_garbage = 0
        filtered_tokens = []
        dropped_tokens = []

        ocr_segs.sort(key=lambda s: s.start)

        kept_ocr = []
        for s in ocr_segs:
            text = s.text.strip()
            # 1. Drop by confidence
            if s.confidence < self.config.ocr_min_confidence:
                dropped_confidence += 1
                dropped_tokens.append(text)
                continue

            # 2. Drop single-character noise unless explicitly allowed
            if len(text) == 1:
                is_digit = text.isdigit()
                is_chinese = bool(re.match(r"^[\u4e00-\u9fff]$", text))
                is_allowed_letter = text.lower() in {"a", "i", "o", "y", "u"}
                if not (is_digit or is_chinese or is_allowed_letter):
                    dropped_length += 1
                    dropped_tokens.append(text)
                    continue

            # 3. Clean uppercase garbage words (acronyms must contain only letters)
            words = text.split()
            cleaned_words = []
            for w in words:
                w_clean = re.sub(r"[.,!?;:()\[\]{}]", "", w)
                if w_clean.isalpha() and w_clean.isupper() and len(w_clean) >= 2:
                    if w_clean in self.config.ocr_allowed_acronyms:
                        cleaned_words.append(w)
                    else:
                        filtered_tokens.append(w_clean)
                else:
                    cleaned_words.append(w)
            
            cleaned_text = " ".join(cleaned_words).strip()
            if not cleaned_text:
                dropped_garbage += 1
                dropped_tokens.append(text)
                continue

            # Update segment text to cleaned text
            s.text = cleaned_text

            # 4. Drop strings with abnormal symbol/letter ratios
            is_number = bool(re.match(r"^\d+([.,]\d+)?$", cleaned_text))
            is_acronym = bool(re.match(r"^[A-Z]{2,}$", cleaned_text))
            if not (is_number or is_acronym):
                normal_chars = 0
                for char in cleaned_text:
                    if char.isalnum() or char.isspace() or char in ".,!?-():;\"'，。！？（）“”：；":
                        normal_chars += 1
                ratio = 1.0 - (normal_chars / max(len(cleaned_text), 1))
                if ratio > self.config.ocr_max_abnormal_char_ratio:
                    dropped_abnormal += 1
                    dropped_tokens.append(cleaned_text)
                    continue

            # 5. Drop repeated OCR text across adjacent frames
            is_duplicate = False
            for prev in reversed(kept_ocr):
                if s.start - prev.start > self.config.ocr_repetition_window:
                    break
                similarity = TextSimilarityService.get_similarity(cleaned_text, prev.text)
                if similarity > 0.8:
                    is_duplicate = True
                    break
            if is_duplicate:
                dropped_repetition += 1
                dropped_tokens.append(cleaned_text)
                continue

            kept_ocr.append(s)

        filtered_segments = kept_ocr + non_ocr_segs
        filtered_segments.sort(key=lambda s: s.start)

        metrics = {
            "ocr_segments_before": before_count,
            "ocr_segments_after": len(kept_ocr),
            "dropped_by_confidence": dropped_confidence,
            "dropped_by_length": dropped_length,
            "dropped_by_abnormal_characters": dropped_abnormal,
            "dropped_by_repetition": dropped_repetition,
            "dropped_by_garbage": dropped_garbage,
            "filtered_tokens": filtered_tokens,
            "dropped_tokens": dropped_tokens,
        }
        return filtered_segments, metrics