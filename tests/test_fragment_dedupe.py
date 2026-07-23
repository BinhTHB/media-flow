"""Tests for STEP-6 fragment deduplication fix."""
import pytest
from unittest.mock import MagicMock, patch
from src.domain.entities.subtitle_region import SubtitleRegion
from src.infrastructure.providers.subtitle.hybrid_subtitle_region_detector_provider import (
    HybridSubtitleRegionDetectorProvider,
)


def _region(start, end, tid, x=0.1, y=0.7, w=0.3, h=0.08):
    return SubtitleRegion(
        start_time=start,
        end_time=end,
        x=x,
        y=y,
        width=w,
        height=h,
        confidence=0.85,
        classification_id=f\"cls:{tid}\",
        is_subtitle=True,
        classifier_model=\"test\",
        track_id=tid,
        proposal_ids=[],
        category=\"dialogue_subtitle\",
        classifier_confidence=0.9,
    )


def test_keeps_confirmed_region_and_adjacent_fragment():
    \"\"\"A confirmed full region should NOT cause an adjacent fragment to be deleted.\"\"\"
    confirmed = _region(0.0, 2.0, \"T001\")  # full region > 0.5s
    fragment = _region(2.1, 2.4, \"T002\")   # fragment <= 0.5s, adjacent but different track
    result = HybridSubtitleRegionDetectorProvider._deduplicate_confirmed_fragments(
        [confirmed, fragment], 1280, 720
    )
    assert len(result) == 2, \"Both should be kept: fragment adjacent to confirmed is fine\"


def test_keeps_two_adjacent_fragments_no_mutual_deletion():
    \"\"\"Two adjacent fragments should NOT delete each other (STEP-6 regression).\"\"\"
    frag_a = _region(0.0, 0.4, \"T001\")
    frag_b = _region(0.5, 0.9, \"T002\")
    result = HybridSubtitleRegionDetectorProvider._deduplicate_confirmed_fragments(
        [frag_a, frag_b], 1280, 720
    )
    assert len(result) == 2, \"Two adjacent fragments must both survive (no mutual deletion)\"


def test_keeps_chain_of_three_fragments():
    \"\"\"A chain of 3 fragments should ALL survive (STEP-6 regression).\"\"\"
    f1 = _region(0.0, 0.4, \"T001\")
    f2 = _region(0.5, 0.9, \"T002\")
    f3 = _region(1.0, 1.3, \"T003\")
    result = HybridSubtitleRegionDetectorProvider._deduplicate_confirmed_fragments(
        [f1, f2, f3], 1280, 720
    )
    assert len(result) == 3, \"Chain of 3 fragments must all survive\"


def test_removes_fragment_duplicating_confirmed_region():
    \"\"\"A fragment that overlaps a confirmed full region AND has same geometry should be removed.\"\"\"
    confirmed = _region(0.0, 2.0, \"T001\", x=0.1, y=0.7, w=0.3, h=0.08)
    duplicate = _region(0.5, 0.9, \"T002\", x=0.1, y=0.7, w=0.3, h=0.08)
    # Same geometry, overlapping time, different track
    result = HybridSubtitleRegionDetectorProvider._deduplicate_confirmed_fragments(
        [confirmed, duplicate], 1280, 720
    )
    assert len(result) == 1, \"Duplicate fragment should be removed\"
    assert result[0].track_id == \"T001\", \"Confirmed region should survive\"


def test_keeps_fragment_with_different_geometry():
    \"\"\"A fragment with different position should NOT be removed.\"\"\"
    confirmed = _region(0.0, 2.0, \"T001\", x=0.1, y=0.7, w=0.3, h=0.08)
    different = _region(0.5, 0.9, \"T002\", x=0.5, y=0.3, w=0.2, h=0.05)  # different position
    result = HybridSubtitleRegionDetectorProvider._deduplicate_confirmed_fragments(
        [confirmed, different], 1280, 720
    )
    assert len(result) == 2, \"Different-geometry fragment should be kept\"

