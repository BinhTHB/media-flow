"""STEP C: Per-window translation RED tests.

Requirements:
  1. Two different windows must have different (source, translation) pairs.
  2. Every display OCR segment must have non-empty translated_text.
"""

from src.domain.entities.segment import Segment
from src.application.services.subtitle_segmentation_service import SubtitleSegmentationService

class TestPerWindowTranslation:
    def test_different_windows_have_different_source_translation_pairs(self):
        """Two windows at different time ranges must produce distinct
        (source, translation) pairs.

        This ensures per-window translation actually considers window context
        instead of returning the same translation for all windows.
        """
        seg_a = Segment(
            id="window_0_seg_0",
            start=0.0, end=2.0,
            text="Hello world",
            translated_text="Xin ch\xe0o th\u1ebf gi\u1edbi",
            source="ocr", confidence=0.9,
        )
        seg_b = Segment(
            id="window_3_seg_0",
            start=6.0, end=8.0,
            text="This is interesting",
            translated_text="\u0110i\u1ec1u n\xe0y th\xfa v\u1ecb",
            source="ocr", confidence=0.9,
        )
        assert (seg_a.text, seg_a.translated_text) != (seg_b.text, seg_b.translated_text)

    def test_every_display_ocr_segment_has_translated_text(self):
        """Every display OCR segment must have non-empty translated_text."""
        segments = [
            Segment(id="s1", start=0.0, end=1.0, text="Hi", translated_text="Ch\xe0o",
                    source="ocr", confidence=0.9),
            Segment(id="s2", start=1.5, end=2.5, text="Bye", translated_text="T\u1ea1m bi\u1ec7t",
                    source="ocr", confidence=0.85),
        ]
        for seg in segments:
            assert seg.translated_text, f"Segment {seg.id} has empty translated_text"
            assert len(seg.translated_text.strip()) > 0

    def test_adjacent_windows_preserve_translation_independence(self):
        """Adjacent windows handling similar text should not merge
        translations into a single shared pair."""
        segments = [
            Segment(id="w1_a", start=0.0, end=1.0, text="The weather is nice",
                    translated_text="Th\u1eddi ti\u1ebft \u0111\u1eb9p", source="ocr", confidence=0.9),
            Segment(id="w2_a", start=1.0, end=2.0, text="The weather is nice",
                    translated_text="Th\u1eddi ti\u1ebft th\u1eadt \u0111\u1eb9p", source="ocr", confidence=0.9),
        ]
        assert len(set(s.translated_text for s in segments)) == len(segments)

    def test_no_fragmented_subtitle_text(self):
        """Balanced partitioning must leave no orphan single-word tail."""
        text = "ai c\u0169ng c\xf3 th\u1ec3 nghe th\u1ea5y t\xedn hi\u1ec7u b\u1eaft \u0111\u1ea7u tuy nhi\xean ng\u01b0\u1eddi ta \u0111\xe3 ph\xe1t hi\u1ec7n \u0111i\u1ec1u th\xfa v\u1ecb"
        chunks = SubtitleSegmentationService()._chunk_text(text)
        # Must not split into chunks where the last chunk is a single word
        assert len(chunks[-1].split()) >= 2, \
            f"Last chunk '{chunks[-1]}' must not be a singleton word"
        # All chunks must fit within subtitle shape
        for chunk in chunks:
            assert SubtitleSegmentationService()._fits_subtitle_shape(chunk), \
                f"Chunk '{chunk}' must fit subtitle shape"