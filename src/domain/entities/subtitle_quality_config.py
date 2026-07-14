from dataclasses import dataclass, field
from typing import List
from src.shared.exceptions import ValidationError

@dataclass
class SubtitleQualityConfig:
    max_chars_per_line: int = 42
    max_lines_per_subtitle: int = 2
    min_duration: float = 0.8
    max_duration: float = 6.0
    merge_gap_threshold: float = 0.3
    deduplicate: bool = True
    prefer_speech_over_ocr: bool = True
    remove_empty_segments: bool = True
    trim_whitespace: bool = True
    ocr_min_confidence: float = 0.70
    ocr_max_abnormal_char_ratio: float = 0.25
    ocr_repetition_window: float = 1.5
    ocr_max_source_leakage: int = 0
    min_coverage_ratio: float = 0.0
    max_longest_gap: float = 999.0
    min_caption_count: int = 0
    ocr_grouping_max_gap: float = 1.5
    ocr_allowed_acronyms: List[str] = field(
        default_factory=lambda: ["AI", "MVP", "FAQ", "USD", "VND", "CEO"]
    )
    enable_content_quality_audit: bool = True
    enable_content_quality_filtering: bool = False
    enable_vietnamese_text_correction: bool = False
    min_caption_quality_score: float = 0.50
    drop_placeholder_captions: bool = True
    drop_malformed_ocr_captions: bool = False
    drop_incomplete_fragments: bool = False

    def __post_init__(self):
        if not isinstance(self.enable_vietnamese_text_correction, bool):
            raise ValidationError("enable_vietnamese_text_correction must be a boolean.")
        if self.max_chars_per_line <= 0:
            raise ValidationError("max_chars_per_line must be positive.")
        if self.max_lines_per_subtitle <= 0:
            raise ValidationError("max_lines_per_subtitle must be positive.")
        if self.min_duration < 0:
            raise ValidationError("min_duration must be non-negative.")
        if self.max_duration < self.min_duration:
            raise ValidationError("max_duration must be greater than or equal to min_duration.")
        if not 0.0 <= self.ocr_min_confidence <= 1.0:
            raise ValidationError("ocr_min_confidence must be between 0.0 and 1.0.")
        if not 0.0 <= self.ocr_max_abnormal_char_ratio <= 1.0:
            raise ValidationError("ocr_max_abnormal_char_ratio must be between 0.0 and 1.0.")
        if self.ocr_repetition_window < 0:
            raise ValidationError("ocr_repetition_window must be non-negative.")
        if self.ocr_max_source_leakage < 0:
            raise ValidationError("ocr_max_source_leakage must be non-negative.")
        if not 0.0 <= self.min_coverage_ratio <= 1.0:
            raise ValidationError("min_coverage_ratio must be between 0.0 and 1.0.")
        if self.max_longest_gap < 0:
            raise ValidationError("max_longest_gap must be non-negative.")
        if self.min_caption_count < 0:
            raise ValidationError("min_caption_count must be non-negative.")
        if self.ocr_grouping_max_gap < 0:
            raise ValidationError("ocr_grouping_max_gap must be non-negative.")
        if not 0.0 <= self.min_caption_quality_score <= 1.0:
            raise ValidationError("min_caption_quality_score must be between 0.0 and 1.0.")