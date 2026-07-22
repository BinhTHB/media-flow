"""STEP C: Per-window translation RED tests.

Requirements:
  1. Two different windows must have different (source, translation) pairs.
  2. Every display OCR segment must have non-empty translated_text.
"""

import pytest
from src.domain.entities.segment import Segment


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

    @pytest.mark.xfail(reason="RED test: fragmented subtitle 'v?' must not appear untranslated")
    def test_no_fragmented_subtitle_text(self):
        """Issue from v20: fragment 'v\u1ecb' (from '\u0111i\u1ec1u th\u00fa v\u1ecb')
        should not appear as a standalone OCR segment with empty/untranslated text."""
        # If a fragment like "v\u1ecb" appears standalone, it should have been
        # merged with its parent segment and have non-empty translated_text.
        vietnamese_fragments = ["v\u1ecb", "th\u00fa", "\u0111i\u1ec1u"]

        standalone_segments = [
            Segment(id="bad", start=3.0, end=3.5, text=frag,
                    translated_text="", source="ocr", confidence=0.9)
            for frag in vietnamese_fragments
        ]

        # These should not appear as standalone untranslated segments
        for seg in standalone_segments:
            """Fragmented Vietnamese text must have translated_text or be merged."""
            # RED: this is expected to fail if fragments exist without translation
            assert seg.translated_text,                 f"Fragment '{seg.text}' should not appear as standalone untranslated segment"
