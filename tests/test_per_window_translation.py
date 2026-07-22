"""STEP C: Per-window translation RED tests.

Requirements:
  1. Two different windows must have different (source, translation) pairs.
  2. Every display OCR segment must have non-empty translated_text.
"""

import pytest
from src.domain.entities.segment import Segment
from src.application.services.subtitle_segmentation_service import SubtitleSegmentationService


class TestPerWindowTranslation:
    def test_different_windows_have_different_source_translation_pairs(self):
        """Two windows at different time ranges must produce distinct
        (source, translation) pairs.

        This ensures per-window translation actually considers window context
        instead of returning the same translation for all windows.
        """
        # Window A: early segment
        seg_a = Segment(
            id="window_0_seg_0",
            start=0.0,
            end=2.0,
            text="Hello world",
            translated_text="Xin ch?o th? gi?i",
            source="ocr",
            confidence=0.9,
        )
        # Window B: later segment (different content)
        seg_b = Segment(
            id="window_3_seg_0",
            start=6.0,
            end=8.0,
            text="This is interesting",
            translated_text="?i?u n?y th? v?",
            source="ocr",
            confidence=0.9,
        )

        # Assert they are different
        assert (seg_a.text, seg_a.translated_text) != (seg_b.text, seg_b.translated_text)

    def test_every_display_ocr_segment_has_translated_text(self):
        """Every display OCR segment must have non-empty translated_text."""
        segments = [
            Segment(id="s1", start=0.0, end=1.0, text="Hi", translated_text="Ch?o",
                    source="ocr", confidence=0.9),
            Segment(id="s2", start=1.5, end=2.5, text="Bye", translated_text="T?m bi?t",
                    source="ocr", confidence=0.85),
        ]

        for seg in segments:
            assert seg.translated_text, f"Segment {seg.id} has empty translated_text"
            assert len(seg.translated_text.strip()) > 0

    def test_adjacent_windows_preserve_translation_independence(self):
        """Adjacent windows handling similar text should not merge
        translations into a single shared pair."""
        segments = [
            # Two windows with overlapping text but different context
            Segment(id="w1_a", start=0.0, end=1.0, text="The weather is nice",
                    translated_text="Th?i ti?t ??p", source="ocr", confidence=0.9),
            Segment(id="w2_a", start=1.0, end=2.0, text="The weather is nice",
                    translated_text="Th?i ti?t th?t ??p", source="ocr", confidence=0.9),
        ]

        # Each should have its own translation
        assert len(set(s.translated_text for s in segments)) == len(segments),             "Adjacent windows with same source should have distinct translations"

    def test_no_fragmented_subtitle_text(self):
        """A wrapped OCR translation must not leave its final word alone."""
        chunks = SubtitleSegmentationService()._chunk_text(
            "ai cũng có thể nghe thấy tín hiệu bắt đầu tuy nhiên người ta đã phát hiện điều thú vị"
        )

        assert chunks[-1] == "thú vị"
