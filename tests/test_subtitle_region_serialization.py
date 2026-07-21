import json
import pytest
from src.domain.entities.subtitle_region import SubtitleRegion


def test_subtitle_region_provenance_round_trip():
    """Provenance fields survive serialization round-trip (to_dict with include_provenance=True)."""
    original = SubtitleRegion(
        start_time=1.0,
        end_time=3.0,
        x=0.1,
        y=0.8,
        width=0.5,
        height=0.1,
        confidence=0.9,
        classification_id="T00001_R0",
        is_subtitle=True,
        classifier_model="gemini-track-verifier-v1",
        track_id="T00001",
        proposal_ids=["F00000_C01", "F00001_C01"],
        category="dialogue_subtitle",
        classifier_confidence=0.97,
    )

    # Without include_provenance: only base 7 fields
    basic = original.to_dict()
    assert set(basic.keys()) == {
        "start_time", "end_time", "x", "y", "width", "height", "confidence"
    }, "to_dict() default must not expose provenance fields"

    # With include_provenance: full 14 fields
    full = original.to_dict(include_provenance=True)
    assert full["classification_id"] == "T00001_R0"
    assert full["is_subtitle"] is True
    assert full["classifier_model"] == "gemini-track-verifier-v1"
    assert full["track_id"] == "T00001"
    assert full["proposal_ids"] == ["F00000_C01", "F00001_C01"]
    assert full["category"] == "dialogue_subtitle"
    assert full["classifier_confidence"] == 0.97

    # Round-trip through JSON
    serialized = json.dumps(full)
    loaded = json.loads(serialized)
    restored = SubtitleRegion(
        start_time=loaded["start_time"],
        end_time=loaded["end_time"],
        x=loaded["x"],
        y=loaded["y"],
        width=loaded["width"],
        height=loaded["height"],
        confidence=loaded["confidence"],
        classification_id=loaded["classification_id"],
        is_subtitle=loaded["is_subtitle"],
        classifier_model=loaded["classifier_model"],
        track_id=loaded["track_id"],
        proposal_ids=loaded["proposal_ids"],
        category=loaded["category"],
        classifier_confidence=loaded["classifier_confidence"],
    )
    assert restored.classification_id == original.classification_id
    assert restored.to_dict(include_provenance=True) == original.to_dict(include_provenance=True)

    # None provenance fields are accepted
    bare = SubtitleRegion(1.0, 3.0, 0.1, 0.8, 0.5, 0.1, 0.9)
    bare_dict = bare.to_dict(include_provenance=True)
    assert bare_dict["classification_id"] is None
    assert bare_dict["is_subtitle"] is None
    assert bare_dict["proposal_ids"] == []
    assert bare_dict["track_id"] is None
