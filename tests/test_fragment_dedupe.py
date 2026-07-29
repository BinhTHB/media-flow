"""Regression coverage for the removed V3 fragment deduplication stage."""

from src.infrastructure.providers.subtitle.hybrid_subtitle_region_detector_provider import (
    HybridSubtitleRegionDetectorProvider,
)


def test_v4_provider_has_no_fragment_deduplication_stage():
    """V4 emits track-scoped geometry and no longer exposes V3 deduplication."""
    assert not hasattr(
        HybridSubtitleRegionDetectorProvider,
        "_deduplicate_confirmed_fragments",
    )

